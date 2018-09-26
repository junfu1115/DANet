import encoding
import shutil

encoding.models.get_model_file('deepten_minc', root='./')
shutil.move('deepten_minc-2e22611a.pth', 'deepten_minc.pth')
