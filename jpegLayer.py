from torch.autograd import Function
import random
from io import BytesIO
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch


def jpeg_compression(images, qf):
    imgsTensor = torch.zeros_like(images)
    for i, image in enumerate(images):
        image = TF.to_pil_image(image.cpu())
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        image_comp = Image.open(outputIoStream)
        imgsTensor[i, :, :, :] = TF.to_tensor(image_comp)

    return imgsTensor.cuda()


class JpegLayerFun(Function):
    @staticmethod
    def forward(ctx, input_, qf):
        ctx.save_for_backward(input_)
        output = jpeg_compression(input_, qf)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input, None


jpegLayer = JpegLayerFun.apply