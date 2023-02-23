import pathlib
import os

parent = r'C:\Users\vishwebh\Desktop\Hackathon\streamlit_app\Images'
for folder in os.listdir(parent):
    fol_path = os.path.join(parent, folder)
    print(fol_path)
    # for file in os.path.list(fol_path):
    #     file_path = os.path.join(fol_path, file)
    path = pathlib.Path(fol_path)
    for img in path.glob('*'):
        print(f'''
              <div class="card">
                <img src="{img.as_posix()}" alt="" />
                <div class="info">
                    <h3>Card Title</h3>
                    <p>
                        Lorem, ipsum dolor sit amet consectetur adipisicing elit. Laborum
                        reiciendis fugit exercitationem quod in officiis minima voluptates
                        eligendi!
                    </p>
                </div>
            </div>
              ''')    

