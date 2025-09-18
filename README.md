Example project training a neural network on the scikit learn digits dataset using [micrograd-rs](https://github.com/mickvangelderen/micrograd-rs).

```bash
cargo run
```

Outputs something like:

```
Loading digits dataset...
Loaded 1796 samples
Train samples: 1436, Test samples: 360
epoch 0, batch 0, loss = 0.9925768783492115
epoch 0, batch 1, loss = 0.9676378469695976
epoch 0, batch 2, loss = 0.9542457917028744
epoch 0, batch 3, loss = 0.9455415058050456
...
epoch 9, batch 85, loss = 0.06145378916137302
epoch 9, batch 86, loss = 0.07913794106701033
epoch 9, batch 87, loss = 0.04830812459705736
epoch 9, batch 88, loss = 0.06885180480321057

Computing predictions on test set...
Test accuracy: 88.89% (320/360 correct)
Saved 360 digit images to test_predictions/
```

Code quality is not great, had claude generate the less exciting parts, but the point was to build something that does a thing, and this does a thing!
