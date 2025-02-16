import os
import cv2
import glob
import vtk

# Funzione per caricare un file OBJ
def load_obj_file(obj_file_path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file_path)
    reader.Update()
    
    mesh = reader.GetOutput()
    if mesh.GetNumberOfCells() == 0:
        print(f"Mesh vuota trovata nel file {obj_file_path}")
        return None
    return mesh
def capture_screenshot(mesh, output_image_path):
    # Crea un mapper per visualizzare la geometria
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    
    # Crea un attore per rappresentare la geometria
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Crea una finestra di rendering
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    
    # Crea una finestra per visualizzare il rendering
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # Imposta una dimensione per la finestra
    render_window.Render()  # Forza il rendering prima di catturare lo screenshot
    
    # Crea un'immagine di output
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()
    
    # Verifica le dimensioni dell'immagine prima di salvarla
    output_image = window_to_image_filter.GetOutput()
    extent = output_image.GetExtent()
    if extent[1] <= 0 or extent[3] <= 0:
        print(f"Errore nel rendering per {output_image_path}: dimensioni non valide.")
        return False

    # Verifica se l'immagine è vuota
    image_array = output_image.GetPointData().GetScalars()
    if image_array is None:
        print(f"Errore nel rendering per {output_image_path}: immagine vuota.")
        return False

    # Salva l'immagine
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_image_path)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()
    
    print(f"Immagine salvata con successo: {output_image_path}")
    return True


def create_video_from_images(frames_jpg_dir, video_path):
    # Ottieni tutte le immagini nella cartella
    images = sorted(glob.glob(os.path.join(frames_jpg_dir, '*.png')))
    if not images:
        print("Errore: nessuna immagine trovata.")
        return

    # Carica la prima immagine per ottenere le dimensioni
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Errore: l'immagine {images[0]} non è valida.")
        return

    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codifica per MP4
    try:
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    except Exception as e:
        print(f"Errore durante l'inizializzazione di VideoWriter: {e}")
        return


    for image_path in images:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Immagine non valida: {image_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video creato con successo: {video_path}")

# Definizione dei percorsi
current_dir = os.path.dirname(os.path.abspath(__file__))
generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")
frames_dir = os.path.join(generated_file_folder_path, "frames")
frames_jpg_dir = os.path.join(generated_file_folder_path, "frames_jpg")
video_path = os.path.join(generated_file_folder_path, "video/output.mp4")

# Crea la cartella per le immagini temporanee
os.makedirs(frames_jpg_dir, exist_ok=True)

# Esegui il rendering di ogni file OBJ e salvalo come immagine
for obj_file in glob.glob(os.path.join(frames_dir, "*.obj")):
    mesh = load_obj_file(obj_file)
    image_path = os.path.join(frames_jpg_dir, os.path.basename(obj_file).replace(".obj", ".png"))
    if not capture_screenshot(mesh, image_path):
        print(f"Errore nel salvataggio dell'immagine per {obj_file}")

# Crea il video con le immagini salvate
create_video_from_images(frames_jpg_dir, video_path)
