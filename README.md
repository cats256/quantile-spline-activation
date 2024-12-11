# quantile-spline-activation
Learnable PyTorch activation function that adapts to input distribution. This activation function is based on linear spline, where locations of control points are determined dynamically based on inputs' distribution quantiles. The inputs' distribution is approximated by storing current and previous forward pass inputs in a sliding window. Works pretty well I'd say. More updates later XD

```python
class CustomActivation(nn.Module):
    def __init__(self, num_features, window_size, num_control_points=9, init_identity=True):
        super(CustomActivation, self).__init__()
        self.i = 0
        self.window_size = window_size
        
        self.data = [torch.tensor(0) for _ in range(window_size)]
        self.quantiles = nn.Parameter(torch.linspace(0, 1, num_control_points + 2)[1:-1], requires_grad=False)
        
        self.a = nn.Parameter(torch.zeros(num_features, num_control_points))
        self.b = nn.Parameter(torch.zeros(num_features, num_control_points))
        
        self.global_bias = nn.Parameter(torch.zeros(1, num_features))
                
        with torch.no_grad():
            if init_identity:
                middle_index = num_control_points // 2
                self.a[:, middle_index] = 1.0
                self.b[:, middle_index] = 1.0

    def forward(self, x):
        if self.training:
            index = self.i % self.window_size
            self.data[index - 1] = self.data[index - 1].detach()            
            self.data[index] = x
            
            all_data = torch.cat(self.data[:min(self.i + 1, self.window_size)], dim=0)
            quantiles = torch.quantile(all_data, self.quantiles, dim=0)
            self.local_bias = quantiles.transpose(0, 1)
            
            self.i += 1
                
        x = x.unsqueeze(-1) + self.local_bias
        x = torch.where(x < 0, self.a * x, self.b * x)
        x = x.sum(dim=-1) + self.global_bias            
        return x
```
