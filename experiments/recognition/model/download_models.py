import encoding
import shutil

encoding.models.get_model_file('deepten_minc', root='./')
shutil.move('deepten_minc-6cb047cd.pth', 'deepten_minc.pth')
