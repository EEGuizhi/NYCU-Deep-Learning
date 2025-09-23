# NYCU IEE Deep Learning Lab 01: Backpropagation
# BSChen (313510156)
from .layer import *

class Network(object):
    r""" Custom Neural Network for Lab 01"""
    def __init__(self):
        # Parameters
        self.is_training = True
        self.show_detail = False

        # Layers
        self.layers = [
            # Convolutional layers
            Conv2d(1, 32, 3, stride=1, padding=1),     # 28x28x1  -> 28x28x32
            BatchNorm2d(32),  SiLU(),
            Conv2d(32, 64, 3, stride=2, padding=1),    # 28x28x32 -> 14x14x64
            BatchNorm2d(64),  SiLU(),
            Conv2d(64, 128, 3, stride=2, padding=1),   # 14x14x64 -> 7x7x128
            BatchNorm2d(128), SiLU(),
            Conv2d(128, 256, 3, stride=2, padding=1),  # 7x7x128  -> 4x4x256
            BatchNorm2d(256), SiLU(),

            # Fully-connected layers
            Flatten(),                   # 4x4x256  -> 4096
            Linear(4096, 1024),          # 4096     -> 1024
            BatchNorm1d(1024), SiLU(),
            Linear(1024, 256),           # 1024     -> 256
            BatchNorm1d(256),  SiLU(),
            Linear(256, 64),             # 256      -> 64
            BatchNorm1d(64),   SiLU(),
            Linear(64, 10)               # 64       -> 10
        ]

        self.out_loss = SoftmaxCrossEntropyLoss(one_hot=True)
        self.center_loss = CenterLoss(num_classes=10, feat_dim=256, lambda_c=0.003)

    def parameters(self) -> list:
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append(layer.weights)
            if hasattr(layer, 'biases'):
                params.append(layer.biases)
            if hasattr(layer, 'gamma'):
                params.append(layer.gamma)
            if hasattr(layer, 'beta'):
                params.append(layer.beta)
        # if hasattr(self.center_loss, 'centers'):
        #     params.append(self.center_loss.centers)
        return params

    def gradients(self) -> list:
        grads = []
        for layer in self.layers:
            if hasattr(layer, 'd_weights'):
                grads.append(layer.d_weights)
            if hasattr(layer, 'd_biases'):
                grads.append(layer.d_biases)
            if hasattr(layer, 'd_gamma'):
                grads.append(layer.d_gamma)
            if hasattr(layer, 'd_beta'):
                grads.append(layer.d_beta)
        # if hasattr(self.center_loss, 'd_centers'):
        #     grads.append(self.center_loss.d_centers)
        return grads

    def forward(self, input: np.ndarray, target: np.ndarray, hook_layer: int=None, hook_arr: np.ndarray=None) -> np.ndarray:
        if self.show_detail:
            print("Forwarding...")

        # Assertions for input and target
        assert len(input.shape) == 4, "Input should be 4-D numpy array"
        assert input.shape[1] == 1, "Input should have 1 channel"
        assert len(target.shape) == 2, "Target should be 2-D numpy array"
        assert input.shape[0] == target.shape[0], "Input and target should have same batch size"

        # Forward pass
        out = input
        for i, layer in enumerate(self.layers):
            # Layer forward
            if isinstance(layer, (BatchNorm2d, BatchNorm1d)):
                out = layer.forward(out, self.is_training)
            else:
                out = layer.forward(out)

            # Center loss forward
            if isinstance(layer, Linear) and layer.out_features == 256:
                center_loss = self.center_loss.forward(out.copy(), np.argmax(target, axis=1))

            # Hook for feature extraction
            if hook_layer is not None and hook_arr is not None and i == hook_layer:
                assert hook_arr.shape == out.shape, "Hook array shape does not match"
                np.copyto(hook_arr, out)

            # Show layer details
            if self.show_detail:
                print(f"Layer {i+1:2d}: {layer.__class__.__name__}, output shape: {out.shape}               ")

        # Output loss
        pred, loss = self.out_loss.forward(out, target)
        if self.show_detail:
            print(f"Layer {len(self.layers):2d}: {self.out_loss.__class__.__name__}, output shape: {pred.shape}               ")

        return pred, loss + center_loss

    def backward(self, backward_center_loss: bool = True):
        if self.show_detail:
            print("Backwarding...")

        # Backward pass
        grad = self.out_loss.backward()
        grad_center = self.center_loss.backward() if hasattr(self, 'center_loss') else 0
        for layer in reversed(self.layers):
            # Center loss backward
            if isinstance(layer, Linear) and layer.out_features == 256:
                grad += grad_center if backward_center_loss else 0

            # Layer backward
            grad = layer.backward(grad)

            # Show layer details
            if self.show_detail:
                print(f"Layer {self.layers.index(layer)+1:2d}: {layer.__class__.__name__}, grad shape: {grad.shape}               ")

    def update(self, learning_rate: float = 1e-3):
        if self.show_detail:
            print("Updating...")

        # Update parameters
        for layer in self.layers:
            layer.update(learning_rate)
        self.center_loss.update(learning_rate)

    def zero_grad(self):
        if self.show_detail:
            print("Zeroing gradients...")

        # Zero gradients
        for layer in self.layers:
            layer.zero_grad()
        self.center_loss.zero_grad()

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def state_dict(self) -> dict:
        state = {}
        for i, layer in enumerate(self.layers):
            state[f'layer_{i}'] = layer.state_dict()
        state['center_loss'] = self.center_loss.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        for i, layer in enumerate(self.layers):
            if f'layer_{i}' in state_dict:
                layer.load_state_dict(state_dict[f'layer_{i}'])
        self.center_loss.load_state_dict(state_dict['center_loss'])



# -------------------------------- Optimizer -------------------------------- #

class Adam:
    def __init__(self, params: list[np.ndarray], lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        `params`: list of parameters (numpy arrays) to optimize
        `lr`: learning rate
        `beta1`, `beta2`: momentum coefficients
        `eps`: small value to avoid division by zero
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # moment estimates
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

        # time step
        self.t = 0

    def step(self, grads: list[np.ndarray]):
        """ Perform one optimization step.
        - `grads`: list of gradients for each parameter (same order as params)
        """
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # update biased second raw moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # update parameter
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, grads: list[np.ndarray]):
        """ Reset gradients to zero (useful between iterations). """
        for g in grads:
            g.fill(0.0)

    def state_dict(self) -> dict:
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'm': self.m,
            'v': self.v,
            't': self.t
        }

    def load_state_dict(self, state_dict: dict):
        self.lr = state_dict['lr']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']


class SGDM:
    def __init__(self, params: list[np.ndarray], lr=1e-2, momentum=0.9):
        """
        `params`: list of parameters (numpy arrays) to optimize
        `lr`: learning rate
        `momentum`: momentum coefficient
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum

        # velocity
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]):
        """ Perform one optimization step.
        - `grads`: list of gradients for each parameter (same order as params)
        """
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # update velocity
            self.v[i] = self.momentum * self.v[i] - self.lr * g
            # update parameter
            self.params[i] += self.v[i]

    def zero_grad(self, grads: list[np.ndarray]):
        """ Reset gradients to zero (useful between iterations). """
        for g in grads:
            g.fill(0.0)

    def state_dict(self) -> dict:
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'v': self.v
        }

    def load_state_dict(self, state_dict: dict):
        self.lr = state_dict['lr']
        self.momentum = state_dict['momentum']
        self.v = state_dict['v']
