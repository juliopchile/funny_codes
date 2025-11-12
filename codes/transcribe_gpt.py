import openai
from openai import OpenAI
import os

from super_secrets import OPENAI_API_KEY

# Configuración de la API de OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)  # Reemplace con su clave API

def transcribir_audio(archivo_audio: str, modelo: str = "gpt-4o-transcribe", idioma: str = "es"):
    """
    Transcribe un archivo de audio utilizando el modelo Whisper de OpenAI.
    
    Parámetros:
        archivo_audio (str): Ruta al archivo de audio (MP3, WAV, M4A, etc.; máx. 25 MB).{
        modelo (str): Modelo a usar (predeterminado: 'whisper-1'). o gpt-4o-transcribe
        idioma (str): Código ISO 639-1 del idioma principal (predeterminado: 'es' para español).
    
    Retorna:
        dict: Objeto con la transcripción completa y segmentos con timestamps.
    """
    if not os.path.exists(archivo_audio):
        raise FileNotFoundError(f"Archivo no encontrado: {archivo_audio}")
    
    with open(archivo_audio, "rb") as f:
        transcripcion = client.audio.transcriptions.create(
            model=modelo,
            file=f,
            #language=idioma,
            #response_format="verbose_json"  # Incluye timestamps y segmentos
            prompt= r'Hay una batalla de freestyle que quiero transcribir pero hay una parte donde no se que dicen: "Así la cierro hermano. Yo si me aferro en el beat vas a perder, de repente me siento [???]. Clavo mil, damos mil porque yo soy Harry Potter, yo me cojo a Hermione." Ayudame a identificar la parte faltante. Utilizando el contexto, que crees que dice?'
        )
    
    return transcripcion

# Ejemplo de uso
if __name__ == "__main__":
    archivo = "inputs/ritter_clip.wav"  # Reemplace con la ruta real de su archivo
    
    try:
        resultado = transcribir_audio(archivo)

        print("Transcripción completa:\n")
        # resultado puede ser un objeto o un dict; intentar ambos
        if isinstance(resultado, dict):
            text = resultado.get("text")
        else:
            text = getattr(resultado, "text", None)
        if text:
            print(text)
        else:
            print("(No se encontró texto en la respuesta)")
        
        print("\nSegmentos con timestamps:\n")
        # obtener segmentos de forma segura desde dict o atributo
        if isinstance(resultado, dict):
            segments = resultado.get("segments")
        else:
            segments = getattr(resultado, "segments", None)

        if segments:
            for seg in segments:
                if isinstance(seg, dict):
                    start = seg.get("start")
                    end = seg.get("end")
                    seg_text = seg.get("text", "").strip()
                else:
                    start = getattr(seg, "start", None)
                    end = getattr(seg, "end", None)
                    seg_text = (getattr(seg, "text", "") or "").strip()

                start_str = f"{float(start):.2f}s" if start is not None else "N/A"
                end_str = f"{float(end):.2f}s" if end is not None else "N/A"
                print(f"[{start_str} - {end_str}] {seg_text}")
        else:
            print("(No se encontraron segmentos con timestamps)")
            
    except Exception as e:
        print(f"Error: {e}")