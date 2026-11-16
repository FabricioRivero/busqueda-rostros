# Búsqueda básica de rostros

API que genera **embeddings faciales** con un modelo local y los guarda en una base vectorial **Chroma** local. Permite registrar personas por su foto y luego identificarlas con otra foto, usando **similitud coseno**.

## Arquitectura

Todo corre en Docker, en dos contenedores Linux:

| Contenedor | Imagen | Puerto | Rol |
|---|---|---|---|
| `bd-chroma` | `chromadb/chroma:0.6.3` | 8000 | Base vectorial Chroma local (datos persistentes en el volumen `chroma-data`) |
| `app-reconocimiento-facial` | build de `./app` (Python 3.11 + FastAPI) | 8081 (host) | API con Swagger que se conecta a `bd-chroma` |

```
foto ──► MTCNN (detecta y recorta rostro 160x160) ──► FaceNet InceptionResnetV1 (VGGFace2) ──► vector 512-d ──► Chroma (coseno)
```

### Modelo

**FaceNet** (`facenet-pytorch`): MTCNN para detección + InceptionResnetV1 preentrenado en VGGFace2. Genera un vector de 512 números normalizado (norma 1). Es distinto a CLIP, el modelo usado en clase: CLIP compara imágenes y texto en general, mientras que FaceNet está entrenado solo para distinguir identidades de rostros.

Los pesos se descargan al construir la imagen, así el contenedor corre sin internet. Todo se ejecuta en CPU.

## Servicios (Swagger)

Documentación interactiva en **http://localhost:8081/docs**.

### `POST /registrar`
`multipart/form-data` con:
- `foto`: imagen con **un solo rostro** (JPG, PNG, WEBP o BMP)
- `nombre`: ej. `Barack Obama`
- `ci`: ej. `4341241`

Genera el embedding y lo guarda en la colección `rostros` con metadatos `{nombre, ci}`. Responde `201`:
```json
{ "mensaje": "Persona registrada", "id": "…", "nombre": "Barack Obama", "ci": "4341241" }
```
Si la foto tiene 0 o más de un rostro responde `422`.

### `POST /buscar`
`multipart/form-data` con `foto`. Genera el embedding, consulta en Chroma el vecino más cercano por coseno y responde:
```json
{
  "encontrado": true,
  "nombre": "Barack Obama",
  "ci": "4341241",
  "metadatos": { "nombre": "Barack Obama", "ci": "4341241" },
  "similitud": 0.78,
  "mensaje": "Persona encontrada"
}
```
Si la similitud es menor a `UMBRAL_SIMILITUD` (0.55 por defecto) responde `encontrado: false`.

**Las imágenes no se guardan**: se procesan en memoria y se descartan. En Chroma solo quedan el embedding, el nombre y el CI.

## Cómo ejecutarlo

Requisito: Docker Desktop (o Docker Engine + Compose).

```bash
docker compose up -d --build
```

La primera vez tarda unos minutos (descarga PyTorch CPU y los pesos del modelo). Luego abrir http://localhost:8081/docs.

### Probar con curl

```bash
curl -F "foto=@obama1.jpg" -F "nombre=Barack Obama" -F "ci=4341241" http://localhost:8081/registrar
curl -F "foto=@obama2.jpg" http://localhost:8081/buscar
```

### Detener

```bash
docker compose down        # conserva los embeddings
docker compose down -v     # borra también el volumen de Chroma
```

## Variables de entorno (`app-reconocimiento-facial`)

| Variable | Default | Descripción |
|---|---|---|
| `CHROMA_HOST` | `bd-chroma` | Host de Chroma |
| `CHROMA_PORT` | `8000` | Puerto de Chroma |
| `CHROMA_COLECCION` | `rostros` | Colección donde se guardan los embeddings |
| `UMBRAL_SIMILITUD` | `0.55` | Similitud coseno mínima para aceptar una coincidencia |

## Estructura

```
.
├── docker-compose.yml
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py        # FastAPI: /registrar y /buscar
    └── rostros.py     # MTCNN + FaceNet -> embedding 512-d
```
