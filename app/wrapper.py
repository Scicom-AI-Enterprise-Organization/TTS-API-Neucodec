"""
https://x.com/mobicham/status/1988992144019775735
https://gist.github.com/mobicham/13ee493ad570b01139b6939b6e13af90
Modified to support external stream specification
"""

import torch

class CUDAGraphsWrapper:
    GRAPH_CACHE = {}
    
    @staticmethod
    def wrap(fn, inputs, kwargs=None, warmup_iters=5, stream=None):
        if kwargs is None:
            kwargs = {}

        def get_key(inputs, stream):
            shapes_key = tuple([tuple(t.shape) + (t.device.index,) for t in inputs if isinstance(t, torch.Tensor)])
            stream_key = id(stream) if stream is not None else id(torch.cuda.current_stream())
            return shapes_key + (stream_key,)

        capture_stream = stream if stream is not None else torch.cuda.current_stream()
        
        inputs_static = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
        outputs_static = None

        torch.cuda.synchronize()
        with torch.cuda.stream(capture_stream), torch.no_grad():
            for _ in range(warmup_iters):
                outputs_static = fn(*inputs_static, **kwargs)
            capture_stream.synchronize()

        run_loop = isinstance(outputs_static, (list, tuple))
        
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream), torch.no_grad():
            graph.capture_begin()
            output = fn(*inputs_static, **kwargs)
            if run_loop:
                for i in range(len(outputs_static)):
                    outputs_static[i].copy_(output[i])
            else:
                outputs_static.copy_(output)
            graph.capture_end()
        
        capture_stream.synchronize()

        key = get_key(inputs_static, capture_stream)
        CUDAGraphsWrapper.GRAPH_CACHE[key] = {
            'inputs': inputs_static,
            'outputs': outputs_static,
            'graph': graph,
            'stream': capture_stream
        }

        @torch.no_grad()
        def fn_with_graph(*inputs, **kwargs):
            key = get_key(inputs, capture_stream)
            cache_entry = CUDAGraphsWrapper.GRAPH_CACHE[key]
            inputs_static = cache_entry['inputs']
            outputs_static = cache_entry['outputs']
            graph = cache_entry['graph']
            cached_stream = cache_entry['stream']
            
            for i in range(len(inputs)):
                if isinstance(inputs[i], torch.Tensor):
                    inputs_static[i].copy_(inputs[i])
            
            with torch.cuda.stream(cached_stream):
                graph.replay()
            
            return outputs_static

        for _ in range(warmup_iters):
            fn_with_graph(*inputs, **kwargs)
        capture_stream.synchronize()

        return fn_with_graph