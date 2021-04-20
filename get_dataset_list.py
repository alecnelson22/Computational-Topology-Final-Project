from pathlib import Path

folder_list = ['UNI_CORR_500', 'BI_CORR', 'CROSSING_90', 'CROSSING_120']

for folder in folder_list:
    for path in Path('./data/' + folder).rglob('*.csv'):
        print('./' + str(path))
