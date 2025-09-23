# NYCU IEE Deep Learning Lab 01: Backpropagation
# BSChen (313510156)
import numpy as np


# ---------------------------- Tool Functions ---------------------------- #


def im2col(
        input: np.ndarray, k_size: int,
        stride: int, padding: int
    ) -> tuple[np.ndarray, int, int]:
    r""" Convert image to column matrix.
    - `input`: (batch_size, in_channels, height, width)
    - `return`: (batch_size, in_channels * k_size * k_size, out_h * out_w)
    """
    batch_size, in_c, in_h, in_w = input.shape
    out_h = (in_h + 2 * padding - k_size) // stride + 1
    out_w = (in_w + 2 * padding - k_size) // stride + 1

    # Padding
    input_padded = np.pad(
        input,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant"
    ) if padding > 0 else input

    cols = np.zeros((batch_size, in_c * k_size * k_size, out_h * out_w))

    col_idx = 0
    for r in range(0, in_h + 2 * padding - k_size + 1, stride):
        for c in range(0, in_w + 2 * padding - k_size + 1, stride):
            patch = input_padded[:, :, r:r+k_size, c:c+k_size]
            cols[:, :, col_idx] = patch.reshape(batch_size, -1)
            col_idx += 1

    return cols, out_h, out_w


def col2im(
        cols: np.ndarray, input_shape: tuple[int, int, int, int],
        k_size: int, stride: int, padding: int
    ) -> np.ndarray:
    r""" Convert column matrix back to image.
    - `input`: (batch_size, out_h * out_w, in_channels * k_size * k_size)
    - `return`: (batch_size, in_channels, height, width)
    """
    batch_size, in_c, in_h, in_w = input_shape
    out_h = (in_h + 2 * padding - k_size) // stride + 1
    out_w = (in_w + 2 * padding - k_size) // stride + 1

    # (batch_size, out_h, out_w, in_c, k_size, k_size)
    cols_reshaped = cols.reshape(batch_size, out_h, out_w, in_c, k_size, k_size)

    # Initialize padded image
    h_padded, w_padded = in_h + 2 * padding, in_w + 2 * padding
    img_padded = np.zeros((batch_size, in_c, h_padded, w_padded), dtype=cols.dtype)

    # Reconstruct image
    for y in range(out_h):
        for x in range(out_w):
            h_start = y * stride
            w_start = x * stride
            img_padded[
                :, :, h_start:h_start+k_size, w_start:w_start+k_size
            ] += cols_reshaped[:, y, x, :, :, :]

    return img_padded if padding == 0 else img_padded[:, :, padding:-padding, padding:-padding]


# ---------------------- Important NN Layers ----------------------------- #


class _Layer(object):
    r""" Layer Template """
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError

    def update(self, lr: float):
        pass

    def zero_grad(self):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass


class Conv2d(_Layer):
    r""" 2D Convolutional Layer """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.biases = np.zeros((out_channels, 1))

        # For backward
        self.input = None
        self.input_shape = None
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray) -> np.ndarray:
        r"""Forward propagation of convolution layer."""
        batch_size, in_c, in_h, in_w = input.shape
        self.input_shape = input.shape
        assert input.shape[1] == self.in_channels, "Input channels do not match."

        # Initialize input
        in_cols, out_h, out_w = im2col(input, self.kernel_size, self.stride, self.padding)
        self.input = in_cols.transpose(0, 2, 1)  # (batch_size, out_h*out_w, in_c*k_size*k_size)
        w_cols = self.weights.reshape(self.out_channels, -1)  # (out_c, in_c*k_size*k_size)

        # Perform convolution as matrix multiplication
        output = w_cols[None, :, :] @ in_cols  # (batch_size, out_c, out_h*out_w)
        output += self.biases[None, :, :]
        output = output.reshape(batch_size, self.out_channels, out_h, out_w)

        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        r"""Backward propagation of convolution layer."""
        batch_size, out_c, out_h, out_w = output_grad.shape
        assert out_c == self.out_channels, "Output channels do not match."

        # (batch_size, out_c, out_h, out_w) -> (batch_size, out_h*out_w, out_c)
        grad = output_grad.transpose(0, 2, 3, 1).reshape(batch_size, -1, out_c)
        self.d_weights = np.sum(
            # (batch_size, out_c, out_h*out_w) @ (batch_size, out_h*out_w, in_c*k_size*k_size)
            (grad.transpose(0, 2, 1) @ self.input), axis=0
        ).reshape(self.weights.shape)  # (out_c, in_c*k_size*k_size) -> (out_c, in_c, k_size, k_size)
        self.d_biases = np.sum(grad, axis=(0, 1)).reshape(self.biases.shape)  # (out_c, 1)

        # (batch_size, out_h*out_w, in_c*k_size*k_size)
        input_grad_cols = grad @ self.weights.reshape(self.out_channels, -1)[None, :, :]
        input_grad = col2im(
            input_grad_cols,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.padding
        )  # (batch_size, in_c, in_h, in_w)

        return input_grad

    def update(self, lr: float):
        self.weights -= lr * self.d_weights
        self.biases -= lr * self.d_biases

    def zero_grad(self):
        self.d_weights.fill(0)
        self.d_biases.fill(0)

    def state_dict(self) -> dict:
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def load_state_dict(self, state_dict: dict):
        self.weights = state_dict['weights']
        self.biases = state_dict['biases']


class Linear(_Layer):
    r""" Fully-connected Layer """
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.randn(out_features, in_features)
        self.biases = np.zeros((out_features,))  # (out_f,)

        self.input = None
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input  # (batch_size, in_f)
        return input @ self.weights.T + self.biases[None, :]  # (batch_size, out_f)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        # gradient w.r.t. input
        input_grad = output_grad @ self.weights  # (batch_size, in_f)

        # gradients w.r.t. weights
        self.d_weights = output_grad.T @ self.input  # (out_f, in_f)
        self.d_biases = np.sum(output_grad, axis=0)  # (out_f,)

        return input_grad

    def update(self, lr: float):
        self.weights -= lr * self.d_weights
        self.biases -= lr * self.d_biases

    def zero_grad(self):
        self.d_weights.fill(0)
        self.d_biases.fill(0)

    def state_dict(self) -> dict:
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def load_state_dict(self, state_dict: dict):
        self.weights = state_dict['weights']
        self.biases = state_dict['biases']


class BatchNorm2d(_Layer):
    r""" 2D Batch Normalization Layer """
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Initialize parameters
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))

        # Running estimates
        self.run_mean = np.zeros((1, num_features, 1, 1))
        self.run_var = np.ones((1, num_features, 1, 1))

        # For backward
        self.input = None
        self.normalized_input = None
        self.batch_mean = None
        self.batch_var = None
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, in_c, in_h, in_w = input.shape
        assert in_c == self.num_features, "Number of features do not match."

        if training:
            self.input = input

            # Compute mean and variance for the batch
            self.batch_mean = np.mean(input, axis=(0, 2, 3), keepdims=True)
            self.batch_var = np.var(input, axis=(0, 2, 3), keepdims=True)

            # Normalize
            self.normalized_input = (input - self.batch_mean) / np.sqrt(self.batch_var + self.eps)

            # Update running estimates
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * self.batch_mean
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * self.batch_var

            output = self.gamma * self.normalized_input + self.beta
            return output
        else:
            # Use running estimates for inference
            normalized_input = (input - self.run_mean) / np.sqrt(self.run_var + self.eps)
            output = self.gamma * normalized_input + self.beta
            return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        batch_size, in_c, in_h, in_w = output_grad.shape
        m = batch_size * in_h * in_w

        # Gradients w.r.t. gamma and beta
        self.d_gamma = np.sum(output_grad * self.normalized_input, axis=(0, 2, 3), keepdims=True)
        self.d_beta = np.sum(output_grad, axis=(0, 2, 3), keepdims=True)

        # Gradient w.r.t. normalized input
        norm_input_grad = output_grad * self.gamma

        # Gradient w.r.t. input
        input_grad = (1. / (m * np.sqrt(self.batch_var + self.eps))) * (
            m * norm_input_grad
            - np.sum(norm_input_grad, axis=(0, 2, 3), keepdims=True)
            - self.normalized_input * np.sum(norm_input_grad * self.normalized_input, axis=(0, 2, 3), keepdims=True)
        )

        return input_grad

    def update(self, lr: float):
        self.gamma -= lr * self.d_gamma
        self.beta -= lr * self.d_beta

    def zero_grad(self):
        self.d_gamma.fill(0)
        self.d_beta.fill(0)

    def state_dict(self) -> dict:
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            'run_mean': self.run_mean,
            'run_var': self.run_var
        }

    def load_state_dict(self, state_dict: dict):
        self.gamma = state_dict['gamma']
        self.beta = state_dict['beta']
        self.run_mean = state_dict['run_mean']
        self.run_var = state_dict['run_var']


class BatchNorm1d(_Layer):
    r""" 1D Batch Normalization Layer """
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Initialize parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        # Running estimates
        self.run_mean = np.zeros((1, num_features))
        self.run_var = np.ones((1, num_features))

        # For backward
        self.input = None
        self.normalized_input = None
        self.batch_mean = None
        self.batch_var = None
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, in_f = input.shape
        assert in_f == self.num_features, "Number of features do not match."

        if training:
            self.input = input

            # Compute mean and variance for the batch
            self.batch_mean = np.mean(input, axis=0, keepdims=True)
            self.batch_var = np.var(input, axis=0, keepdims=True)

            # Normalize
            self.normalized_input = (input - self.batch_mean) / np.sqrt(self.batch_var + self.eps)

            # Update running estimates
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * self.batch_mean
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * self.batch_var

            output = self.gamma * self.normalized_input + self.beta
            return output
        else:
            # Use running estimates for inference
            normalized_input = (input - self.run_mean) / np.sqrt(self.run_var + self.eps)
            output = self.gamma * normalized_input + self.beta
            return output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        batch_size, in_f = output_grad.shape
        m = batch_size

        # Gradients w.r.t. gamma and beta
        self.d_gamma = np.sum(output_grad * self.normalized_input, axis=0, keepdims=True)
        self.d_beta = np.sum(output_grad, axis=0, keepdims=True)

        # Gradient w.r.t. normalized input
        norm_input_grad = output_grad * self.gamma

        # Gradient w.r.t. input
        input_grad = (1. / (m * np.sqrt(self.batch_var + self.eps))) * (
            m * norm_input_grad
            - np.sum(norm_input_grad, axis=0, keepdims=True)
            - self.normalized_input * np.sum(norm_input_grad * self.normalized_input, axis=0, keepdims=True)
        )

        return input_grad

    def update(self, lr: float):
        self.gamma -= lr * self.d_gamma
        self.beta -= lr * self.d_beta

    def zero_grad(self):
        self.d_gamma.fill(0)
        self.d_beta.fill(0)

    def state_dict(self) -> dict:
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            'run_mean': self.run_mean,
            'run_var': self.run_var
        }

    def load_state_dict(self, state_dict: dict):
        self.gamma = state_dict['gamma']
        self.beta = state_dict['beta']
        self.run_mean = state_dict['run_mean']
        self.run_var = state_dict['run_var']


# ------------------------- Activation Functions ------------------------- #

class ReLU(_Layer):
    r"""ReLU Activation Layer"""
    def __init__(self):
        self.input = None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        input_grad = output_grad * (self.input > 0)
        return input_grad


class ELU(_Layer):
    r"""ELU Activation Layer"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.input = None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.where(input > 0, input, self.alpha * (np.exp(input) - 1))

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        input_grad = output_grad * np.where(self.input > 0, 1, self.alpha * np.exp(self.input))
        return input_grad


class SiLU(_Layer):
    r"""SiLU (Swish) Activation Layer"""
    def __init__(self):
        self.input = None
        self.sigmoid_x = None

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.sigmoid_x = self.sigmoid(input)
        return input * self.sigmoid_x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = self.sigmoid_x + self.input * self.sigmoid_x * (1 - self.sigmoid_x)
        return grad_output * grad_input


# --------------------------- Other Functions --------------------------- #


class MaxPool2d(_Layer):
    r"""2D Max Pooling Layer"""
    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.argmax_mask = None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        batch_size, in_c, in_h, in_w = input.shape
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, in_c, out_h, out_w))

        # mask to store argmax
        self.argmax_mask = np.zeros_like(input)

        for i in range(out_h):
            for j in range(out_w):
                h_st = i * self.stride
                h_ed = h_st + self.kernel_size
                w_st = j * self.stride
                w_ed = w_st + self.kernel_size

                window = input[:, :, h_st:h_ed, w_st:w_ed]
                max_val = np.max(window, axis=(2, 3), keepdims=True)
                mask = (window == max_val)
                self.argmax_mask[:, :, h_st:h_ed, w_st:w_ed] += mask

                output[:, :, i, j] = max_val.squeeze(-1).squeeze(-1)

        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        out_h, out_w = output_grad.shape[2], output_grad.shape[3]
        input_grad = np.zeros_like(self.input)

        for i in range(out_h):
            for j in range(out_w):
                h_st = i * self.stride
                h_ed = h_st + self.kernel_size
                w_st = j * self.stride
                w_ed = w_st + self.kernel_size

                mask = self.argmax_mask[:, :, h_st:h_ed, w_st:w_ed]
                grad = output_grad[:, :, i, j][:, :, None, None]  # (B, C, 1, 1)
                input_grad[:, :, h_st:h_ed, w_st:w_ed] += mask * grad

        return input_grad


class Flatten(_Layer):
    r"""Flatten Layer"""
    def __init__(self):
        self.input_shape = None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input_shape = input.shape
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)  # (N, C, H, W) -> (N, C*H*W)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.reshape(self.input_shape)  # (N, C*H*W) -> (N, C, H, W)


# ---------------------------- Loss Functions ---------------------------- #


class SoftmaxCrossEntropyLoss(_Layer):
    r"""Softmax with Cross Entropy Loss Layer"""
    def __init__(self, one_hot: bool = False):
        self.logits = None
        self.targets = None
        self.prob = None
        self.loss = None
        self.one_hot = one_hot

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
        # Store logits and targets
        self.logits = logits
        self.targets = targets

        # Compute softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Compute loss
        if self.one_hot:
            self.loss = -np.mean(np.sum(targets * np.log(self.prob + 1e-9), axis=1))
        else:
            self.loss = -np.mean(np.log(self.prob[np.arange(logits.shape[0]), targets] + 1e-9))

        return self.prob, self.loss

    def backward(self) -> np.ndarray:
        input_grad = self.prob.copy()
        if self.one_hot:
            input_grad -= self.targets
        else:
            input_grad[np.arange(self.prob.shape[0]), self.targets] -= 1
        input_grad /= self.prob.shape[0]
        return input_grad

    def update(self, lr: float):
        pass

    def zero_grad(self):
        self.logits = None
        self.targets = None
        self.prob = None
        self.loss = None


class CenterLoss(_Layer):
    r""" Center Loss Layer """
    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 0.003, momentum: float = 0.1):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        # Initialize centers
        self.centers = np.random.randn(num_classes, feat_dim)
        self.momentum = momentum

        # For backward
        self.features = None
        self.labels = None
        self.loss = None
        self.d_centers = np.zeros_like(self.centers)

    def forward(self, features: np.ndarray, labels: np.ndarray) -> float:
        r""" Forward propagation of center loss.
        - `features`: (batch_size, feat_dim)
        - `labels`: (batch_size,)
        """
        self.features = features  # (batch_size, feat_dim)
        self.labels = labels      # (batch_size,)

        batch_size = features.shape[0]
        centers_batch = self.centers[labels]  # (batch_size, feat_dim)
        self.loss = self.lambda_c * np.mean(np.sum((features - centers_batch) ** 2, axis=1))

        return self.loss

    def backward(self) -> np.ndarray:
        batch_size = self.features.shape[0]
        centers_batch = self.centers[self.labels]  # (batch_size, feat_dim)

        # Gradient w.r.t. features
        input_grad = self.lambda_c * 2 * (self.features - centers_batch) / batch_size  # (batch_size, feat_dim)

        # Gradient w.r.t. centers (not used)
        # for i in range(self.num_classes):
        #     mask = (self.labels == i)
        #     if np.any(mask):
        #         diff = self.centers[i] - self.features[mask]
        #         self.d_centers[i] = 2 * self.lambda_c * np.sum(diff, axis=0) / (np.sum(mask))
        #     else:
        #         self.d_centers[i] = 0

        # Moving average update for centers
        self.d_centers.fill(0)
        for i in range(self.num_classes):
            indices = (self.labels == i)
            if np.any(indices):
                self.d_centers[i] = self.centers[i] - self.features[indices].mean(axis=0)

        return input_grad

    def update(self, lr: float = None):
        self.centers -= self.d_centers * self.momentum

    def zero_grad(self):
        self.d_centers.fill(0)

    def state_dict(self) -> dict:
        return {
            'centers': self.centers,
            'lambda_c': self.lambda_c,
            'momentum': self.momentum
        }

    def load_state_dict(self, state_dict: dict):
        self.centers = state_dict['centers']
        self.lambda_c = state_dict['lambda_c']
        self.momentum = state_dict['momentum'] if 'momentum' in state_dict else 0.01
