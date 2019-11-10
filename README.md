# Multi_meadia

1. Before run this system, there are several parameters needed to be double checked, in  `data_processing.py`:
    * `DSpath = 'DataSet/LabeledDB'`, indicating your database root path
    * `refinedPath = 'DataSet/RefinedMeshes',`indicating where to store refined meshed
    * `cleanOff_jar = 'cleanoff.jar'`, indicating the path of tool used to refine meshed, already in the git
    * `cleanMesh= 'refined_mesh.txt'`, indicating the path of a txt file used to store path of refined mesh, this file would be automatically created once the path is set
    * `cleanOFFListPathtxt = 'issue_final.txt'`, indicating the path of a txt file used to store path of poorly-sampled mesh, this file would be automatically created once the path is set

1. To initiate this 3D shape retrieval system, you can simply run `python main.py` on terminal, so that you can select a target mesh (OFF or PLY format) to query similar items back from the system.
2. All the functions (e.g mesh normalization, feature extraction etc.)related to preprocess dataset are stored in `data_processing.py`.
3. LPSB dataset is stored in `'DataSet/LabeledDB'`
4. The extracted features are saved in CSV format, could be found in `csvFiles` folder

