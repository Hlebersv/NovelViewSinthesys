import torch
from typing import *
from torch import nn
from einops import rearrange
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPVisionConfig, CLIPPreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput, CLIPVisionTransformer


class UNet(UNet2DConditionModel):
    def __init__(self, path='lambdalabs/sd-image-variations-diffusers', subfolder='unet', revision='v2.0'):
        super(UNet, self).__init__()
        self.path = path
        self.subfolder = subfolder
        self.revision = revision
        self.base_model = UNet2DConditionModel.from_pretrained(self.path, subfolder=self.subfolder,
                                                               revision=self.revision)
        self.fc = nn.Linear(in_features=770, out_features=768)

    def forward(self, x, t, image, diff_angles, **kwargs):
        # diff_angles is of shape [b, 1, 2]
        # image is of shape [b, 1, 768]

        # reshape the inputs to be two-dimensional
        diff_angles = rearrange(diff_angles, 'b l c -> b (l c)')  # diff_angles is now of shape [b, 2]
        image = rearrange(image, 'b l c -> b (l c)')  # image is now of shape [b, 768]

        # concatenate the inputs along the last dimension
        c = torch.cat((image, diff_angles), dim=-1)  # c is now of shape [b, 770]

        # pass through a fully connected layer
        c = self.fc(c)  # c is now of shape [b, 768]

        # reshape c to match the expected input shape of the base_model
        c = rearrange(c, 'b c -> b 1 c')  # c is now of shape [b, 1, 768]

        output = self.base_model(x, t, c)
        return output


class CLIPVision(CLIPPreTrainedModel): # for future upd

    """
    Realisation from transformers library, see https://github.com/huggingface/transformers/blob/f1660d7e23d4432513fe060bde4f9b7b29f05204/src/transformers/models/clip/modeling_clip.py#L76
    """
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionTransformer(config)

        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)

        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )




