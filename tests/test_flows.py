import unittest
import jax.numpy as jnp
import numpy as np
import jax

from models.normalizing_flow.core import NormalizingFlowDist

key = jax.random.PRNGKey(0)

class TestRealNVP(unittest.TestCase):

    def test_is_forward_bijection(self):
        from models.normalizing_flow.bijections import RealNVP
        from models.normalizing_flow.nn import MLP
        dim = 2
        net = MLP([32,32], output_dim=2)
        rnvp = RealNVP(net, False)
        params = rnvp.init(key, jnp.ones((1,2)))
            
        u0 = jnp.array(np.random.normal(size=(4, dim)))

        v = rnvp.apply(params, u0, method=rnvp.forward)[0]
        u1 = rnvp.apply(params, v, method=rnvp.backward)[0]

        assert jnp.allclose(jnp.round(u0, 4), jnp.round(u1, 4))

    def test_shift_and_log_scale(self):  
        from models.normalizing_flow.bijections import RealNVP
        from models.normalizing_flow.nn import MLP
        dim = 2
        net = MLP([32,32], output_dim=2)
        rnvp = RealNVP(net, False)        

        u0 = jnp.array(np.random.normal(size=(4, dim // 2)))
        params = rnvp.init(key, jnp.ones((32,2)))
        s = rnvp.apply(params, u0,  method=rnvp.shift_and_log_scale_fn)
        
        assert s[0].shape == u0.shape
        assert s[1].shape == u0.shape


class TestNormalizingFlow(unittest.TestCase):

    def test_is_forward_bijection(self):
        from models.normalizing_flow.bijections import RealNVP
        from models.normalizing_flow.nn import MLP
        from models.normalizing_flow.core import NormalizingFlow
        dim = 2
        def get_rnvp():
            net = MLP([32,32], output_dim=2)
            rnvp = RealNVP(net, False)
            return rnvp
        flow = NormalizingFlow(transforms=[get_rnvp(), get_rnvp()])
        params = flow.init(key, jnp.ones((1,2)))
            
        u0 = jnp.array(np.random.normal(size=(4, dim)))

        v = flow.apply(params, u0, method=flow.forward)[0]
        u1 = flow.apply(params, v, method=flow.backward)[0]

        assert jnp.allclose(jnp.round(u0, 4), jnp.round(u1, 4))



class TestNormalizingFlowDist(unittest.TestCase):

    def test_sample(self):
        from models.normalizing_flow.bijections import RealNVP
        from models.normalizing_flow.nn import MLP
        from models.normalizing_flow.core import NormalizingFlow
        from models.normalizing_flow.distributions import StandardGaussian
        
        dim = 2
        def get_rnvp():
            net = MLP([32,32], output_dim=2)
            rnvp = RealNVP(net, False)
            return rnvp
        flow = NormalizingFlow(transforms=[get_rnvp(), get_rnvp()])
        gaussian = StandardGaussian(2)
        flow_dist = NormalizingFlowDist(gaussian, flow)

        params = flow_dist.init(key, jnp.ones((1,2)), method=flow_dist.log_prob)
        samples = flow_dist.apply(params, key, 10, method=flow_dist.sample)
        
        assert samples.shape == (10, 2)