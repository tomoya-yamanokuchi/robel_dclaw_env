

class Normalize:
    def normalize(self, X, x_min, x_max):
        # 最大値を 1、最小値を 0 にする正規化
        return (X - x_min) / (x_max - x_min)

    def denormalize(self, X_normalized, x_min, x_max):
        # 最大値を 1、最小値を 0 にする正規化
        return X_normalized * (x_max - x_min) + x_min
