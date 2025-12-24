import torch
from torch.optim.optimizer import Optimizer

class AutoTrustGO(Optimizer):
    """
    AutoTrustGO（最終安定版）

    【基本コンセプト】
    - 勾配の「信頼できる部分」だけを、連続的に強調して更新する
    - step 依存なし（学習段階は内部統計から自律的に判断）
    - trust_level により、ユーザーが「攻め／守り」を直感的に調整可能
    - SDXL LoRA（画風・キャラ・概念・実写）を想定した汎用設計

    【AdamW / Lion との違い】
    - 勾配ノルム一括制御ではなく「rank × 安定性」で局所判断
    - 微細表現（目・線・質感）を潰しにくい
    - 教師画像への過剰追従や AI 臭さを抑制
    """

    def __init__(self, params, lr=1e-3, trust_level=0.4, eps=1e-8):
        """
        【引数】

        lr : float
            学習率（ベース値）
            推奨:
              - SDXL LoRA 画風学習: 1.0e-4 ～ 2.0e-4
              - キャラ / 概念学習: 8.0e-5 ～ 1.5e-4
              - 実写 / 微調整   : 5.0e-5 ～ 1.0e-4

        trust_level : float [0.0 - 1.0]
            この optimizer 唯一の「性格調整パラメータ」

            低いほど:
              - 更新が慎重
              - ノイズ耐性が高い
              - 学習は遅めだが破綻しにくい

            高いほど:
              - 更新が積極的
              - 収束が速い
              - 攻めた学習が可能（過学習リスク増）

            推奨:
              - 画風学習      : 0.35 ～ 0.45
              - キャラ学習    : 0.40 ～ 0.55
              - 概念学習      : 0.30 ～ 0.45
              - 実写 / 微調整 : 0.25 ～ 0.40

        eps : float
            ゼロ割防止用の微小値（通常は変更不要）
        """

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        # ユーザー操作パラメータ（必ず 0-1 に正規化）
        self.trust_level = float(max(0.0, min(1.0, trust_level)))

        # Momentum 係数（AdamW よりやや軽め）
        self.beta = 0.94

        self.eps = eps

        # rank 判定用しきい値
        # trust_level が高いほど、rank 条件を厳しくする
        self.thr = 0.60 - 0.25 * self.trust_level

        # --- 学習状態の EMA ---
        # mask_ema : 有効更新割合の安定度
        # flip_ema : 勾配符号反転の頻度（不安定さ指標）
        self.mask_ema = 0.0
        self.flip_ema = 0.0

        # consensus（符号安定性）用の履歴上限
        self.consensus_cap = int(2 + round(2 * self.trust_level))

    @torch.no_grad()
    def step(self, closure=None):
        """
        1 step の更新処理
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 統計集計用
        total_mask = 0.0
        total_flip = 0.0
        total_elems = 0.0

        for group in self.param_groups:
            base_lr = group["lr"]

            # ==================================================
            # 学習率調整（flip のみを見る）
            # ==================================================
            # 勾配の符号が頻繁に反転している = 不安定
            x = max(0.0, min(1.0, self.flip_ema))

            # trust_level が高いほど、flip に敏感に減速
            lr_scale = 1.0 - (0.15 + 0.15 * self.trust_level) * x
            lr_scale = max(0.6, min(1.0, lr_scale))

            lr = base_lr * lr_scale

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # --- 状態初期化 ---
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)
                if "consensus" not in state:
                    state["consensus"] = torch.zeros_like(g)
                if "prev_sign" not in state:
                    state["prev_sign"] = torch.zeros_like(g, dtype=torch.int8)

                # ==================================================
                # flip（符号反転）の検出
                # ==================================================
                sign_g = torch.sign(g).to(torch.int8)
                flip = sign_g != state["prev_sign"]
                total_flip += flip.float().mean().item()
                state["prev_sign"].copy_(sign_g)

                # ==================================================
                # rank（相対勾配強度）
                # ==================================================
                mag = g.abs()
                mag_mean = mag.mean()
                rank = mag / (mag_mean + self.eps)

                # ==================================================
                # confidence 設計（2成分）
                # ==================================================

                # ① rank 成分：大きい勾配ほど信頼
                conf_rank = rank / (rank + 1.0)
                conf_rank = conf_rank.clamp(max=0.85)

                # ② stability 成分：符号が安定しているほど信頼
                state["consensus"].add_(sign_g.float() * conf_rank)
                state["consensus"].clamp_(
                    -self.consensus_cap, self.consensus_cap
                )

                conf_stab = state["consensus"].abs() / self.consensus_cap
                conf_stab = conf_stab.clamp(max=1.0)

                # 最終 confidence
                confidence = conf_rank * (0.6 + 0.4 * conf_stab)

                # 不安定時（flip 多発）は confidence を抑制
                confidence *= (1.0 - 0.3 * self.flip_ema)

                # --- confidence 高止まりの自然減衰 ---
                high_conf = (confidence > 0.6).float().mean().item()
                decay = 1.0 - 0.02 * high_conf * self.flip_ema
                state["consensus"].mul_(decay)

                # ==================================================
                # mask（rank 専用・連続）
                # ==================================================
                # thr 周辺を sigmoid で滑らかに遷移
                rank_gate = torch.sigmoid(
                    (rank - self.thr) / (0.15 + 0.2 * (1.0 - self.mask_ema))
                )

                # mask 下限（学習後期ほど小さくなる）
                floor = 0.08 + 0.22 * (1.0 - self.mask_ema)

                mask = floor + (1.0 - floor) * rank_gate

                mask_ratio = mask.mean().item()
                total_mask += mask_ratio
                total_elems += 1.0

                # ==================================================
                # momentum + 更新量
                # ==================================================
                m = state["momentum"]
                m.mul_(self.beta).add_(g)

                d = (g + self.beta * m) * mask
                norm = d.norm()

                # ==================================================
                # danger（全体ブレーキ）
                # ==================================================
                # ・更新量が大きい
                # ・flip が多い
                # → 危険度が上昇
                danger = (
                    0.6 * (norm / (norm + 1.0)) +
                    0.4 * self.flip_ema
                )

                # ユーザー制御（trust_level）
                local_trust = self.trust_level * (0.4 + 0.4 * mask_ratio)

                # 非線形ブレーキ
                scale = 1.0 / (1.0 + danger * local_trust * norm)
                d.mul_(scale)

                # パラメータ更新
                p.add_(-lr * d)

        # ==================================================
        # EMA 更新（学習段階を自律推定）
        # ==================================================
        if total_elems > 0:
            mask_ratio = total_mask / total_elems
            flip_ratio = total_flip / total_elems

            self.mask_ema = 0.92 * self.mask_ema + 0.08 * mask_ratio
            self.flip_ema = 0.92 * self.flip_ema + 0.08 * flip_ratio

            # rank 閾値の自己調整
            if self.mask_ema < 0.10:
                self.thr *= 0.995
            elif self.mask_ema > 0.35:
                self.thr *= 1.005

            self.thr = float(max(0.3, min(0.7, self.thr)))

        return loss
