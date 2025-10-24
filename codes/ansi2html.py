# requirements ansi2html
# pip install ansi2html

import os
import re
from ansi2html import Ansi2HTMLConverter
from ansi2html.style import SCHEME

def convert_ansi_to_latex(text, dark_mode=True):
    """
    Convierte secuencias /textcolor{<clases>}{<contenido>} con clases de estilo ansi (por ejemplo, "ansi2 ansi32") en comandos LaTeX aplicando el formato adecuado.
    Reglas implementadas (según la lista CSS):
    • ansi1: negrita      --> /textbf{...}
    • ansi2: texto tenue  --> modificar el color a "color!50" (mediante xcolor)
    • ansi3: cursiva      --> /textit{...}
    • ansi4: subrayado    --> /underline{...}
    • ansi5/ansi6: blink  --> se ignoran (se eliminan del estilo)
    • ansi8: hidden       --> se elimina el contenido (cadena vacía)
    • ansi9: tachado      --> /sout{...} (requiere /usepackage[normalem]{ulem})
    """
    # Esta expresión regular busca: \textcolor{<contenido>}{<contenido>}
    # Se asume que no hay anidamiento complicado de llaves.
    pattern = re.compile(r'\\textcolor\{([^}]+)\}\{([^}]*)\}')

    def replacer(match):
        # Extrae las "clases" y el contenido
        classes = match.group(1).split()   # Ej.: ["ansi2", "ansi32"]
        content = match.group(2)
        # Diccionario para los efectos (estilos) mapeados
        effects = {
            'ansi1': 'bold',       # Negrita
            'ansi2': 'dim',        # Dim (interpretado según dark_mode)
            'ansi3': 'italic',     # Cursiva
            'ansi4': 'underline',  # Subrayado
            'ansi5': 'blink',      # Blink, se ignora
            'ansi6': 'blink',      # Blink, se ignora
            'ansi8': 'hidden',     # Oculto
            'ansi9': 'linethrough' # Tachado
        }
        # Se separan las clases que definen efecto de la que define color.
        style_ops = []   # Efectos encontrados (por ejemplo, bold, italic, etc.)
        color = None
        for cls in classes:
            if cls in effects:
                style_ops.append(effects[cls])
            elif re.match(r'ansi\d+', cls):
                # Se espera que la clase de color sea de formato "ansiXX"
                color = cls
        # Si no se especifica color, se usa "black" por defecto.
        if color is None:
            color = 'black'

        # Para aplicar "dim" según dark_mode:
        if 'dim' in style_ops:
            if dark_mode:
                # Cuando se está en dark mode, se oscurece el color mezclándolo con negro.
                color = f"{color}!65!black"
            else:
                # En modo claro se mezcla con blanco para aclarar.
                color = f"{color}!50"
            style_ops.remove('dim')
        
        # Si se aplica "hidden", se omite el contenido.
        if 'hidden' in style_ops:
            return ""
        
        # Se arma el comando base de color.
        result = f"\\textcolor{{{color}}}{{{content}}}"
        
        # Se envuelven de afuera hacia adentro según los estilos
        if 'bold' in style_ops:
            result = f"\\textbf{{{result}}}"
        if 'italic' in style_ops:
            result = f"\\textit{{{result}}}"
        if 'underline' in style_ops:
            result = f"\\underline{{{result}}}"
        if 'linethrough' in style_ops:
            result = f"\\sout{{{result}}}"
        # Se ignoran los efectos "blink"
        
        return result

    # Se aplica el reemplazo globalmente y se regresa el nuevo texto.
    return pattern.sub(replacer, text)


if __name__ == "__main__":
    ansi_file = "iperf3.txt"
    scheme_name = "ansi2html"
    scheme = SCHEME[scheme_name]

    # Leer el archivo de texto con la data en ANSI
    with open(ansi_file, 'r') as text_file:
        ansi_data = text_file.read()

    # Convertir el texto con ANSI en LaTex
    conv = Ansi2HTMLConverter(latex=True)
    latex_data = conv.convert(ansi_data)

    nuevo_texto = convert_ansi_to_latex(latex_data)

    # Printear definición de colores
    for i, n in enumerate(scheme):
        print("\\definecolor{ansi3%s}{HTML}{%s}" % (i, n.strip('#').upper()))
    print()

    # Printear el código LaTex
    print(nuevo_texto)
