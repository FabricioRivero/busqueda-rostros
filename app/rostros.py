"""Generacion de embeddings faciales con FaceNet (facenet-pytorch).

- MTCNN detecta y recorta el rostro (160x160).
- InceptionResnetV1 preentrenado en VGGFace2 genera un vector de 512 numeros.
El modelo corre local (CPU) dentro del contenedor; los pesos se descargan
al construir la imagen Docker.
"""
import io

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps

_dispositivo = torch.device("cpu")
_detector = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=_dispositivo)
_modelo = InceptionResnetV1(pretrained="vggface2").eval().to(_dispositivo)

UMBRAL_DETECCION = 0.90


class ErrorRostro(ValueError):
    """La imagen no tiene un rostro valido para procesar."""


def abrir_imagen(datos: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(datos))
        img = ImageOps.exif_transpose(img)  # respeta la orientacion de fotos de celular
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ErrorRostro("El archivo no es una imagen valida.") from exc


def generar_embedding(datos: bytes, exigir_un_rostro: bool = True) -> list[float]:
    """Devuelve el embedding (512 floats, norma 1) del rostro de la imagen.

    Si exigir_un_rostro es True, falla cuando hay 0 o mas de 1 rostro.
    Si es False, usa el rostro con mayor probabilidad.
    """
    img = abrir_imagen(datos)
    with torch.no_grad():
        caras, probs = _detector(img, return_prob=True)

    if caras is None:
        raise ErrorRostro("No se detecto ningun rostro en la imagen.")

    validos = [i for i, p in enumerate(probs) if p is not None and p >= UMBRAL_DETECCION]
    if not validos:
        raise ErrorRostro("No se detecto ningun rostro con suficiente confianza.")
    if exigir_un_rostro and len(validos) > 1:
        raise ErrorRostro(f"Se detectaron {len(validos)} rostros; la foto debe tener uno solo.")

    mejor = max(validos, key=lambda i: probs[i])
    with torch.no_grad():
        vec = _modelo(caras[mejor].unsqueeze(0).to(_dispositivo))
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec[0].cpu().tolist()
