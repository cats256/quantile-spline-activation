# Adaptive Quantile Activation (AQUA): A learnable activation function that dynamically adapts to input distribution

AQUA is a learnable activation function that adapts to the input distribution as it evolves during training. This activation leverages a linear spline-based approach, where the quantiles of the input distribution dynamically determine the control points' locations during training. The input distribution and quantiles are approximated using a sliding window, or circular buffer, that stores the current and previous forward pass inputs.

Works pretty well I'd say. More updates later. Give the repo a star if you like this XD.

```python
class CustomActivation(nn.Module):
    def __init__(self, num_features, buffer_size=10, num_control_points=9, init_identity=True):
        super(CustomActivation, self).__init__()
        self.i = 0
        self.buffer_size = buffer_size
        
        self.input_buffer = [torch.tensor(0) for _ in range(buffer_size)]
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
            index = self.i % self.buffer_size
            self.input_buffer[index - 1] = self.input_buffer[index - 1].detach()            
            self.input_buffer[index] = x
            
            all_inputs = torch.cat(self.input_buffer[:min(self.i + 1, self.buffer_size)], dim=0)
            quantiles_values = torch.quantile(all_inputs, self.quantiles, dim=0)
            self.local_bias = quantiles_values.transpose(0, 1)
            
            self.i += 1
                
        x = x.unsqueeze(-1) + self.local_bias
        x = torch.where(x < 0, self.a * x, self.b * x)
        x = x.sum(dim=-1) + self.global_bias            
        return x
```
