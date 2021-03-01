import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

columns = ['Topt', 'PS1zmag', 'PS1ymag', 'Jmag', 'W1mag', 'W2mag', 'W3mag']
dwarfs_df = pd.read_csv('Dwarf_Star_Data\dwarf_catalogue.txt', sep=';', usecols=columns, comment='#').apply(pd.to_numeric, errors='coerce')
dwarfs_df = dwarfs_df[~dwarfs_df['Topt'].between(0, 10)].drop(columns='Topt').dropna(how='any', axis=0) 
dwarfs_df['Object'] = 0 #The target column: 0 for dwarfs, 1 for quasars.

quasars_df = pd.read_csv('Quasar_Data\quasars.txt', sep='\s+', usecols=columns[1:])
quasars_df['W1mag'] = quasars_df['W1mag'] - 1.7  #   Changing Units
quasars_df['W2mag'] = quasars_df['W2mag'] - 3.34 #
quasars_df['Object'] = 1

objects_df = dwarfs_df.append(quasars_df, ignore_index=True) #The dataset is imbalanced, with quasars only 11% of the dataset.

X = objects_df.drop(columns=['Object'])
y = objects_df['Object']

X['W1-W2'] = X.W1mag-X.W2mag      #
X['W2-W3'] = X.W2mag-X.W3mag      #    Defining the colours used as the features.
X['z-y'] = X.PS1zmag-X.PS1ymag    #    
X['y-W1'] = X.PS1ymag-X.W1mag     #
X['y-W2'] = X.PS1ymag-X.W2mag     #
X['y-J'] = X.PS1ymag-X.Jmag       #
X['J-W1'] = X.Jmag-X.W1mag        #
X['J-W2'] = X.Jmag-X.W2mag        #

X = X.drop(columns=columns[1:])   #Removing the original data from the features.

new_data = pd.read_csv('New_Data\new_data_colours.csv', sep=',')
coordinates = new_data[['ra', 'dec']]
X_new = new_data.drop(columns=['ra', 'dec'])

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

over = SMOTE(sampling_strategy=0.2, random_state=random_state)
under = RandomUnderSampler(sampling_strategy=0.6, random_state=random_state)

tree = DecisionTreeClassifier(min_samples_leaf = 0.1, random_state=random_state)
pipeline = Pipeline([('over', over), ('under', under), ('tree', tree)])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

y_pred_new = pipeline.predict(X_new)
y_pred_quasar_coords = coordinates.values[np.where(y_pred_new==1)]
y_pred_dwarf_coords = coordinates.values[np.where(y_pred_new==0)]

np.set_printoptions(suppress=True, threshold=np.inf)

print('\nNo. of Quasars Predicted in the Holdout Set: {}'.format(y_pred.sum()))
print('No. of Quasars in the Holdout Set: {}'.format(y_test.sum()))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

quasar_coords = pd.read_csv('Quasar_Data\catalogue.txt', sep=',', usecols=['ra', 'dec']).values

quasars_found_no = 0
quasars_found_coords = []

quasars_missed_no = 0
quasars_missed_coords = []

suspected_quasar_coords = []

for coord in quasar_coords:
	for pred_coord in y_pred_quasar_coords:
		if (coord[0]-pred_coord[0])**2 + (coord[1]-pred_coord[1])**2 < (4/3600):
			quasars_found_no += 1
			quasars_found_coords.append(pred_coord)		

	for pred_coord in y_pred_dwarf_coords:
		if (coord[0]-pred_coord[0])**2 + (coord[1]-pred_coord[1])**2 < (4/3600):
			quasars_missed_no += 1
			quasars_missed_coords.append(pred_coord)

print('Quasars Predicted in the New Data: {}'.format(y_pred_new.sum()))
print('Pre-Discovered Quasars Found: {}'.format(quasars_found_no))
print('Pre-Discovered Quasars Missed: {}'.format(quasars_missed_no))
print('\nCoordinates of Discovered Quasars: \n\n  RA            DEC\n{}'.format(np.array(quasars_found_coords)))
print('\nCoordinates of Missed Quasars: \n\n  RA            DEC\n{}'.format(np.array(quasars_missed_coords)))
print('\nCoordinates of Predicted Quasars: \n\n  RA            DEC\n{}'.format(np.array(y_pred_quasar_coords)))