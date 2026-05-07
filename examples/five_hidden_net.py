import jax

import matnets as mtn


class FiveHiddenNet:
    def __init__(
        self,
        key,
        *,
        input_neurons=3,
        hidden_neurons=4,
        output_neurons=1,
        n=2,
    ):
        keys = jax.random.split(key, 6)
        self.n = n
        self.hidden_layers = 5
        self.params = {
            "hidden": [
                mtn.init(keys[0], p=input_neurons, q=hidden_neurons, n=n),
                mtn.init(keys[1], p=hidden_neurons, q=hidden_neurons, n=n),
                mtn.init(keys[2], p=hidden_neurons, q=hidden_neurons, n=n),
                mtn.init(keys[3], p=hidden_neurons, q=hidden_neurons, n=n),
                mtn.init(keys[4], p=hidden_neurons, q=hidden_neurons, n=n),
            ],
            "out": mtn.init(keys[5], p=hidden_neurons, q=output_neurons, n=n),
        }

    def __call__(self, x):
        return self.forward(self.params, x)

    @staticmethod
    def forward(params, x):
        for layer in params["hidden"]:
            x = mtn.dense(layer, x, activation=jax.nn.relu)
        return mtn.dense(params["out"], x)

    def jit(self):
        return jax.jit(self.forward)


def main():
    input_neurons = 3

    keys = jax.random.split(jax.random.key(42), 2)
    model = FiveHiddenNet(keys[0], input_neurons=input_neurons, n=2)
    x = jax.random.normal(keys[1], (input_neurons, model.n, model.n))

    compiled_forward = model.jit()
    y = compiled_forward(model.params, x)

    print("input:", x.shape)
    print("hidden layers:", model.hidden_layers)
    print("n:", model.n)
    print("output:", y.shape)
    print(y)


if __name__ == "__main__":
    main()
