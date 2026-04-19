from dotenv import load_dotenv
from roboflow import Roboflow
import os


load_dotenv()

# Получаем ключ из переменных окружения
api_key = os.getenv("ROBOFLOW_API_KEY")
workspace = os.getenv("ROBOFLOW_WORKSPACE")
project_name = os.getenv("cell-detection")
version = int(os.getenv("ROBOFLOW_VERSION", 1))


rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace).project(project_name)
dataset = project.version(2).download(model_format="yolov8")

workspace = rf.workspace()
print(f"Ваш workspace: {workspace}")

# Список всех проектов в workspace
projects = workspace.list_projects()
print("Ваши проекты:")
for project in projects:
    print(f"  - {project.name} (id: {project.id})")