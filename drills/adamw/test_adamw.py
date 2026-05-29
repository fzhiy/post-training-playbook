import unittest
import torch
import torch.nn as nn
from from_scratch import AdamWFromScratch

class TestAdamWFromScratch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    def test_basic_adamw(self):
        """Test basic AdamW optimization with default parameters."""
        # Create a simple model
        model = nn.Linear(10, 5)
        model.to(self.device)
        
        # Reference model with PyTorch AdamW
        ref_model = nn.Linear(10, 5)
        ref_model.load_state_dict(model.state_dict())
        ref_model.to(self.device)
        
        # Create optimizers with same hyperparameters
        optimizer = AdamWFromScratch(model.parameters(), lr=0.01, weight_decay=0.01)
        ref_optimizer = torch.optim.AdamW(ref_model.parameters(), lr=0.01, weight_decay=0.01)
        
        # Generate random data
        x = torch.randn(3, 10, device=self.device)
        y = torch.randn(3, 5, device=self.device)
        
        # Run several steps
        for _ in range(5):
            # Zero gradients
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            ref_output = ref_model(x)
            
            # Compute loss
            loss = nn.MSELoss()(output, y)
            ref_loss = nn.MSELoss()(ref_output, y)
            
            # Backward pass
            loss.backward()
            ref_loss.backward()
            
            # Check gradients are finite
            for param in model.parameters():
                self.assertTrue(torch.isfinite(param.grad).all(), 
                              "Found non-finite gradient in from-scratch optimizer")
            
            # Optimizer step
            optimizer.step()
            ref_optimizer.step()
        
        # Check that parameters match after optimization
        for (name, param), (ref_name, ref_param) in zip(
            model.named_parameters(), ref_model.named_parameters()
        ):
            self.assertEqual(param.shape, ref_param.shape, 
                           f"Shape mismatch for {name}")
            self.assertTrue(torch.allclose(param, ref_param, atol=1e-5), 
                          f"Parameter mismatch for {name}")
    
    def test_with_bias_correction(self):
        """Test bias correction in Adam optimizer."""
        # Use a single parameter for simplicity
        param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        grad = torch.tensor([0.1, 0.2, 0.3])
        
        # Manually compute Adam update for verification
        beta1, beta2 = 0.9, 0.999
        lr = 0.01
        eps = 1e-8
        
        m = torch.zeros_like(param)
        v = torch.zeros_like(param)
        
        # First step
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad.pow(2)
        
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        
        expected = param - lr * m_hat / (v_hat.sqrt() + eps)
        
        # Create optimizer and simulate one step
        optimizer = AdamWFromScratch([param], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=0)
        param.grad = grad
        optimizer.step()
        
        self.assertTrue(torch.allclose(param, expected, atol=1e-5),
                       "Adam bias correction failed")
        self.assertEqual(param.shape, expected.shape, "Shape mismatch in bias correction test")
        self.assertTrue(torch.isfinite(param).all(), "Non-finite values after optimization")
    
    def test_weight_decay(self):
        """Test decoupled weight decay."""
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        grad = torch.tensor([0.1, 0.1])
        
        lr = 0.01
        weight_decay = 0.1
        
        # Manually compute update: first Adam step without weight decay
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        m = torch.zeros_like(param)
        v = torch.zeros_like(param)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad.pow(2)
        
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        
        # Adam update without weight decay
        param_after_adam = param - lr * m_hat / (v_hat.sqrt() + eps)
        # Then apply weight decay
        expected = param_after_adam - lr * weight_decay * param
        
        # Create optimizer
        optimizer = AdamWFromScratch([param], lr=lr, weight_decay=weight_decay)
        param.grad = grad
        optimizer.step()
        
        self.assertTrue(torch.allclose(param, expected, atol=1e-5),
                       "Weight decay test failed")
        self.assertEqual(param.shape, expected.shape, "Shape mismatch in weight decay test")
        self.assertTrue(torch.isfinite(param).all(), "Non-finite values after weight decay")
    
    def test_maximize_flag(self):
        """Test maximize flag (gradient negation)."""
        param = torch.tensor([1.0], requires_grad=True)
        grad = torch.tensor([0.5])
        
        lr = 0.1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Compute expected update with negated gradient
        m = torch.zeros_like(param)
        v = torch.zeros_like(param)
        
        neg_grad = -grad  # Because maximize=True
        
        m = beta1 * m + (1 - beta1) * neg_grad
        v = beta2 * v + (1 - beta2) * neg_grad.pow(2)
        
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        
        expected = param - lr * m_hat / (v_hat.sqrt() + eps)
        
        # Create optimizer with maximize=True
        optimizer = AdamWFromScratch([param], lr=lr, betas=(beta1, beta2), 
                                    eps=eps, weight_decay=0, maximize=True)
        param.grad = grad
        optimizer.step()
        
        self.assertTrue(torch.allclose(param, expected, atol=1e-5),
                       "Maximize flag test failed")
        self.assertEqual(param.shape, expected.shape, "Shape mismatch in maximize test")
        self.assertTrue(torch.isfinite(param).all(), "Non-finite values after maximize")
    
    def test_different_shapes(self):
        """Test that optimizer handles different parameter shapes correctly."""
        # Create parameters with various shapes
        param1 = torch.randn(5, 10)
        param2 = torch.randn(3,)
        param3 = torch.randn(2, 3, 4)
        
        params = [
            {'params': [param1], 'lr': 0.01},
            {'params': [param2, param3], 'lr': 0.005}
        ]
        
        optimizer = AdamWFromScratch(params, weight_decay=0.01)
        
        # Initialize gradients
        param1.grad = torch.randn_like(param1)
        param2.grad = torch.randn_like(param2)
        param3.grad = torch.randn_like(param3)
        
        # Store initial shapes
        shapes = [p.shape for p in [param1, param2, param3]]
        
        # Run optimizer step
        optimizer.step()
        
        # Check shapes haven't changed
        new_shapes = [p.shape for p in [param1, param2, param3]]
        self.assertEqual(shapes, new_shapes, "Parameter shapes changed after optimization")
        
        # Check all parameters are finite
        self.assertTrue(torch.isfinite(param1).all(), "Non-finite values in param1")
        self.assertTrue(torch.isfinite(param2).all(), "Non-finite values in param2")
        self.assertTrue(torch.isfinite(param3).all(), "Non-finite values in param3")
    
    def test_gradient_finiteness_check(self):
        """Test that optimizer handles non-finite gradients appropriately."""
        param = torch.tensor([1.0], requires_grad=True)
        
        # Create gradient with NaN
        param.grad = torch.tensor([float('nan')])
        optimizer = AdamWFromScratch([param])
        
        # The optimizer should proceed but param will become NaN
        optimizer.step()
        self.assertTrue(torch.isnan(param).any(), "Optimizer should propagate NaN gradients")
        
        # Test with finite gradient after NaN
        param = torch.tensor([1.0], requires_grad=True)
        param.grad = torch.tensor([1.0])
        optimizer = AdamWFromScratch([param])
        optimizer.step()
        self.assertTrue(torch.isfinite(param).all(), 
                       "Optimizer should work with finite gradients")

if __name__ == "__main__":
    unittest.main()
