"""API de busqueda basica de rostros (FastAPI + FaceNet + Chroma).

Solo dos servicios, documentados en Swagger (/docs):
  POST /registrar  -> guarda el embedding del rostro + nombre y CI en Chroma.
  POST /buscar     -> devuelve la persona con el embedding mas cercano (coseno).
Las imagenes NUNCA se guardan: se procesan en memoria y se descartan.
"""
import os
import uuid
from typing import Optional

import chromadb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from rostros import ErrorRostro, generar_embedding

CHROMA_HOST = os.getenv("CHROMA_HOST", "bd-chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLECCION = os.getenv("CHROMA_COLECCION", "rostros")
# Similitud coseno minima para aceptar una coincidencia (FaceNet: misma persona ~0.6+)
UMBRAL_SIMILITUD = float(os.getenv("UMBRAL_SIMILITUD", "0.55"))
TIPOS_PERMITIDOS = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(
    title="API de Busqueda Basica de Rostros",
    description=(
        "Genera embeddings faciales con **FaceNet** (MTCNN + InceptionResnetV1, VGGFace2, 512 dimensiones) "
        "y los guarda en la base vectorial **Chroma** local. La busqueda usa **similitud coseno**.\n\n"
        "Las imagenes no se almacenan: solo el embedding, el nombre y el CI de la persona."
    ),
    version="1.0.0",
)

_coleccion = None


def coleccion():
    """Conexion perezosa a Chroma (espera a que el contenedor bd-chroma este listo)."""
    global _coleccion
    if _coleccion is None:
        cliente = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _coleccion = cliente.get_or_create_collection(
            name=COLECCION, metadata={"hnsw:space": "cosine"}
        )
    return _coleccion


class PersonaRegistrada(BaseModel):
    mensaje: str = Field(examples=["Persona registrada"])
    id: str
    nombre: str = Field(examples=["Barack Obama"])
    ci: str = Field(examples=["4341241"])


class ResultadoBusqueda(BaseModel):
    encontrado: bool
    nombre: Optional[str] = Field(default=None, examples=["Barack Obama"])
    ci: Optional[str] = Field(default=None, examples=["4341241"])
    metadatos: Optional[dict] = None
    similitud: Optional[float] = Field(default=None, description="Similitud coseno (1 = identico)")
    mensaje: str


async def _leer_imagen(foto: UploadFile) -> bytes:
    if foto.content_type not in TIPOS_PERMITIDOS:
        raise HTTPException(415, f"Tipo de archivo no soportado: {foto.content_type}. Use JPG, PNG, WEBP o BMP.")
    datos = await foto.read()
    if not datos:
        raise HTTPException(400, "El archivo esta vacio.")
    return datos


async def _embedding(datos: bytes, exigir_un_rostro: bool) -> list[float]:
    try:
        return await run_in_threadpool(generar_embedding, datos, exigir_un_rostro)
    except ErrorRostro as exc:
        raise HTTPException(422, str(exc)) from exc


def _chroma():
    try:
        return coleccion()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"No se pudo conectar a Chroma en {CHROMA_HOST}:{CHROMA_PORT}") from exc


@app.post(
    "/registrar",
    response_model=PersonaRegistrada,
    status_code=201,
    summary="Registrar una persona",
    description="Recibe una foto con **un solo rostro**, genera su embedding y lo guarda en Chroma con el nombre y CI. La foto no se guarda.",
    tags=["Rostros"],
)
async def registrar(
    foto: UploadFile = File(..., description="Foto con un solo rostro"),
    nombre: str = Form(..., examples=["Barack Obama"]),
    ci: str = Form(..., examples=["4341241"]),
):
    nombre, ci = nombre.strip(), ci.strip()
    if not nombre or not ci:
        raise HTTPException(400, "nombre y ci son obligatorios.")
    vec = await _embedding(await _leer_imagen(foto), exigir_un_rostro=True)
    id_ = str(uuid.uuid4())
    _chroma().add(ids=[id_], embeddings=[vec], metadatas=[{"nombre": nombre, "ci": ci}])
    return PersonaRegistrada(mensaje="Persona registrada", id=id_, nombre=nombre, ci=ci)


@app.post(
    "/buscar",
    response_model=ResultadoBusqueda,
    summary="Buscar a una persona por su rostro",
    description="Recibe una foto, genera su embedding y busca en Chroma el embedding mas cercano por similitud coseno. Devuelve nombre y metadatos.",
    tags=["Rostros"],
)
async def buscar(foto: UploadFile = File(..., description="Foto del rostro a buscar")):
    vec = await _embedding(await _leer_imagen(foto), exigir_un_rostro=False)
    col = _chroma()
    if col.count() == 0:
        return ResultadoBusqueda(encontrado=False, mensaje="No hay personas registradas.")
    r = col.query(query_embeddings=[vec], n_results=1, include=["metadatas", "distances"])
    meta, dist = r["metadatas"][0][0], r["distances"][0][0]
    similitud = round(1 - dist, 4)  # Chroma con hnsw:space=cosine devuelve distancia = 1 - coseno
    if similitud < UMBRAL_SIMILITUD:
        return ResultadoBusqueda(
            encontrado=False, similitud=similitud,
            mensaje="No se encontro una persona suficientemente parecida.",
        )
    return ResultadoBusqueda(
        encontrado=True, nombre=meta.get("nombre"), ci=meta.get("ci"),
        metadatos=meta, similitud=similitud, mensaje="Persona encontrada",
    )
