"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Create the NTM components
        print(f"Creating EncapsulatedNTM with num_inputs={num_inputs}, num_outputs={num_outputs}, controller_size={controller_size}, controller_layers={controller_layers}, num_heads={num_heads}, N={N}, M={M}")
        memory = NTMMemory(N, M)
        print(f"Created memory with N={N}, M={M}")
        controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        print(f"Created controller with num_inputs={num_inputs}, M={M}, num_heads={num_heads}, controller_size={controller_size}, controller_layers={controller_layers}")
        heads = []
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]
        heads = nn.ModuleList(heads)
        print(f"Created heads with num_heads={num_heads}")
        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        print(f"Created NTM with num_inputs={num_inputs}, num_outputs={num_outputs}, controller={controller}, memory={memory}, heads={heads}")
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs).to("cuda:0")

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
