import torch
class HorizontalStripeAugmentation(torch.nn.Module):
    def __init__(self, num_parts, p_mix):
        super().__init__()
        self.num_parts = num_parts
        self.p_mix = p_mix

    def forward(self, input1, input2):
            B, C, H, W = input1.shape
            S = self.num_parts

            #Augment visible images
            mix_prob = torch.rand((B,S))
            is_mix = (mix_prob < self.p_mix).float()
            is_mix = is_mix.unsqueeze(1).repeat(1, C, 1)

            mix_channel = torch.randint(0, C, (B, ))
            mix_channel_onehot = torch.nn.functional.one_hot(mix_channel, num_classes=C)
            mask = mix_channel_onehot[:,:,None] * is_mix

            alpha = torch.distributions.Beta(2, 2).sample((B, C, S))
            alpha = mask * alpha
            x1 = (1-alpha)[:,:,:,None,None] * input1.reshape(B,C,S,H//S,W) + (alpha)[:,:,:,None,None] * input2.reshape(B,C,S,H//S,W)
            x1 = x1.reshape(B,C,H,W)

            #Augment infrared images
            mix_prob = torch.rand((B,S))
            is_mix = (mix_prob < self.p_mix).float()
            is_mix = is_mix.unsqueeze(1).repeat(1, C, 1)

            mix_channel = torch.randint(0, C, (B, ))
            mix_channel_onehot = torch.nn.functional.one_hot(mix_channel, num_classes=C)
            mask = is_mix

            alpha = torch.distributions.Beta(2, 2).sample((B, S)).unsqueeze(1).repeat(1, C, 1)
            alpha = mask * alpha

            input1_gray = (mix_channel_onehot[:,:,None,None] * input1).sum(1)
            input1_gray = input1_gray.unsqueeze(1).repeat(1, C, 1, 1)
            x2 = (1-alpha)[:,:,:,None,None] * input2.reshape(B,C,S,H//S,W) + (alpha)[:,:,:,None,None] * input1_gray.reshape(B,C,S,H//S,W)
            x2 = x2.reshape(B,C,H,W)

            return x1, x2