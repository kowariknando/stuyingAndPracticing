'''
Created on Jun 2, 2020
@author: kowarika
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#todo
dataflair_index = pd.date_range('1/1/2000', periods=8)
print('Dictionaries')
'''##DICTIONARIES
Los dictionaries tienen la estructura {key:value, key2:value2,...}
Para acceder a los values se pone my_dict([key]) y te devuelve el valor del key
Tambien se puede acceder a los values con my_dict.get(key) que si no encuentra el key devuelve None en vez de error
my_dict_estudiando = {1: 'apple', 2: 'ball'}
print(my_dict_estudiando[2]) #esto me da el valor del key 2 es decir Ball
print(my_dict_estudiando.get('ejemplodeKey')) #esto devuelve None, si pusieses  my_dict_estudiando.get('ejemplodeKey') tendría ERROR
my_dict_mixed_keys = {'name': 'John', 1: [2, 4, 3]} # dictionary with mixed keys
print(my_dict_mixed_keys[1])
#emptydicctionary = {} #Initializing an empty dictionary is like initializing an empty list, but with curly braces {} instead of brackets [].
my_dict_estudiando [2]= 'ring' # esto cambia el value of the key 2 to a new value Ring
my_dict_estudiando[3]= 'diabolo' #esto añade un valor nuevo al diccionario diabolo bajo el key 3
print(my_dict_estudiando)
removido = my_dict_estudiando.pop(1) #esto quita el key 1 del dictionary y devuelve el value de 1, es decir apple
print(my_dict_estudiando)
print( 'hemos removido el value: '+ removido)
'''
print('funcion simple def algo(x, y)..')
'''x = 5
y = 10
def funcionEjemplo(a, b):
    print(a + b)
funcionEjemplo(x, y)
'''
print('Tuplas')
## list() LISTAS
my_list_int = [100, 2, 3]
my_list_things = ['primera', 'lista', 'de', 'cosas']
my_list_things2 = ['segunda', 'lista2', 'de2', 'Cosas2']
my_list_things3 = ['tercera', 'lista3', 'de3', 'Cosas3']
print("ESta es mi lista")
print(my_list_int)

print("FINAL DEL EJEMPLOOOOOOOOOOOOO")

small_value = iter(range(3))
print(next(small_value))
print(next(small_value))
print(next(small_value))

values = range(101)
print(values)

values_list = list(values)
print(values_list)
values_sum = sum(values)
print(values_sum)

print('Enumarte(), Zip() y unzip with *z')
'''## enumarate() , zip() UNZIP with *z
for index1, value1 in enumerate(my_list_things):  # for index1, value1 in enumerate(my_list_things, start=1):
    print(index1, value1)
### EJEMPLO DE ZIP()
#ZIP te junta dos listas y te genera una clase/objetoZip que podemos convertir en una lista con list(nombredelzip) y printearla
#tambien podemos convertir el zip en un dictionary: dict(zip(my_list_to_be_zipped, my_list_to_be_zipped2)
datos_para_zip = list(zip(my_list_things, my_list_things2, my_list_things3))
print("esto es el zip printeado es una tupla")
print(datos_para_zip)
print("esto es pasando por todo el zip")
for value1, value2, value3 in zip(my_list_things, my_list_things2, my_list_things3):
    print(value1, value2, value3)
# Unzip print(*datos_para_zip)  #
print("esto es un unzip directo")
print(*datos_para_zip)
print("esto es un unzip de cada fila")
z1 = zip(my_list_things, my_list_things2, my_list_things3)
result1, result2, result3 = zip(*z1)
print(result1)
print(result2)
print(result3)
'''
print('# Import from Internet from a URL and save it on a csv  and draw some of the data')
'''# Import from Internet from a URL and save it on a csv
# Import package
 from urllib.request import urlretrieve
 Import pandas ya lo tngo mas arriba import pandas as pd
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv' # Assign url of file: url
urlretrieve(url, 'winequality-red.csv') # Save file locally
df = pd.read_csv('winequality-red.csv', sep=';')  # Read file into a DataFrame and print its head
 la separaciön en el csv file is ;
print(df.head())
pd.DataFrame.hist(df.head())  # Plot first column of df
 se necesita import matplotlib.pyplot as plt
#TODO #el orginial ponia pd.DataFrame.hist(df.ix[:, 0:1]) pero no me funciona!
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()
'''
print('import the urls from a specific url')
'''from urllib.request import urlopen, Request # Import packages
url = "http://www.datacamp.com/teach/documentation" # Specify the url
request = Request(url) # This packages the request: request
response = urlopen(request) # Sends the request and catches the response: response
print(type(response)) # Print the datatype of response
response.close() # Be polite and close the response!'''
print('Extracting the HTML text from a URL')
'''######Extracting the HTML text from a URL##
import requests
from bs4 import BeautifulSoup # Import packages
url = 'https://www.python.org/~guido/' # Specify url: url
r = requests.get(url) # Package the request, send the request and catch the response: r
html_doc = r.text # Extracts the response as html: html_doc
soup = BeautifulSoup(html_doc) # Create a BeautifulSoup object from the HTML: soup
pretty_soup = soup.prettify() # Prettify the BeautifulSoup object: pretty_soup
print(pretty_soup)# Print the response
print('Import some datas from a URL (online movide DataBase) and print them. Using the API of Online Movie DB.')'''
'''### Import some datas from a URL (online movide DataBase) and print them. Using the API of Online Movie DB.
import requests # Import package
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=social+network' # Assign URL to variable: url
r = requests.get(url) # Package the request, send the request and catch the response: r
json_data = r.json() # Decode the JSON data into a dictionary: json_data
for k in json_data.keys(): # Print each key-value pair in json_data
    print(k + ': ', json_data[k])'''
print('list comprehension')
'''doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']
# Cómo se escribe en teoria:
#[expression for item in list] or  [output expression for iterator variable in iterable]
# [output expression + conditional on output for iterator variable in iterable + conditional on interable] 
squares = [i**2 for i in range(10)]
print(squares)
#for nested lists of comprehension in theory is that:
# [[output expression] for iterator variable in iterable]
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for raw in range(5)]
for row in matrix: # Print the matrix
    print(row)

print('Conditionals in comprehensions')
# conditional comprehension in the input expression
cuadradoSielnumeroesPar = [num**2 for num in range(10) if num % 2 == 0]
print(cuadradoSielnumeroesPar)
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli'] # Create a list of strings: fellowship
new_fellowship = [ member for member in fellowship if len(member)>=7 ] # Create list comprehension: new_fellowship with 7 or more characters
# conditional comprehension in the output expression
ceroImparesCuadradoPares = [num**2 if num%2== 0 else 0 for num in range(10)]
print(ceroImparesCuadradoPares)
new_fellowship2 = [member if len(member)>=7 else "" for member in fellowship] #fellas with more than 7 characters and "" if less

print('Dictionary comprehensions ord Dict Comprehensions') #we use {} instead of [] and the Key and thevalue are separated by :
#esto crea una clase dictionary con los keys del 0 al 1 y con los values el negativo de cada uno
pos_neg = {num: num*10 for num in range(10)}
print(pos_neg)
print(type(pos_neg))
print(pos_neg[4])
new_dictornary_fellowship = { member : len(member) for member in fellowship}
print (new_dictornary_fellowship) #dictionary with the members of the list as the keys and the length of each string as the corresponding values
'''
print('Generators')
ejemplodecomprehension= [2 *num for num in range(5)]
print(ejemplodecomprehension)
#una list comprehension se puede "transformar" en un generator simplemente poniendole ()
generator_ejemplo = (2*num for num in range(5))
print(type(generator_ejemplo))
#List comprehension VS GENERATORS
#Generators doesnt return a list, generators returns a generator object
#BOTH can be iterated over.
for num in generator_ejemplo:
    print(num)  #se puede iterar sobre un generator con un for pero no crea una lista, hay como que ejecturala o llamarla
#como es iterable como una lista se le puede pasar la funcion list para mostrar los elementos del generator
#print(list(generator_ejemplo)) #si lo ejecuto o lo llamo dos veces me da fallo o se queda como vacio no se porque;...
#TODO estudiar porque se queda vacio el generator despues de usarlo una vez.
#print(next(generator_ejemplo)) #se puede ir pasando de elemento en elemento de la "lista" que crea
#print(next(generator_ejemplo))#se puede ir pasando de elemento en elemento de la "lista" que crea
#print(next(generator_ejemplo)) # lo pongo en comentario porque si no me da fallo, elTODO de mas arriba

#it is called lazy evaluation, as'i que puedes hacer algo como en el aire para que no se ejecute hasta que no lo llames
#puedes hacer algo qeue necesita mucha memoria pero sin almacenarlo
generator_con_numero_enorme = (num for num in range(10**1000))
print(type(generator_con_numero_enorme))
even_nums = (num for num in range(10) if num%2 ==0)
print(list(even_nums))
#GENERATOR FUNCTIONS
#instead of return values using RETURN they use YIELD to yield sequences of values
def num_secuencia_generator(n):  #"genera valores del 0 al n"
    i = 0
    while i < n:
        yield i
        i += 1

usando_secuencia_generator = num_secuencia_generator(5)
print(type(usando_secuencia_generator)) #produce un generator object class
for item in usando_secuencia_generator:
    print(item)
#other example of Generator function:
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey'] # Create a list of strings
def get_lengths(input_list): # Define generator function get_lengths
    """Generator function that yields the
    length of the strings in input_list."""
    for person in input_list: # Yield the length of a string
        yield len(person)
for value in get_lengths(lannister): # Print the values generated by get_lengths()
    print(value)

print('Funcion to process a large dataset in chunks')
'''Funcion to process a large dataset in chunks
###### This is the last leg. You've learned a lot about processing a large dataset in chunks. In this last exercise, you will put all the code for processing the data into a single function so that you can reuse the code without having to rewrite the same things all over again.
#####
#####You're going to define the function plot_pop() which takes two arguments: the filename of the file to be processed, and the country code of the rows you want to process in the dataset.
#####
#####Because all of the previous code you've written in the previous exercises will be housed in plot_pop(), calling the function already does the following:
#####
#####Loading of the file chunk by chunk,
#####Creating the new column of urban population values, and
#####Plotting the urban population data.
#####That's a lot of work, but the function now makes it convenient to repeat the same process for whatever file and country code you want to process and visualize!You're going to use the data from 'ind_pop_data.csv', available in your current directory. The packages pandas and matplotlib.pyplot has been imported as pd and plt respectively for your use.

def plot_pop(filename, country_code): # Define plot_pop()
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)  # Initialize reader object: urb_pop_reader
    data = pd.DataFrame() # Initialize empty DataFrame: data
    for df_urb_pop in urb_pop_reader: # Iterate over each DataFrame chunk
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code] # Check out specific country: df_pop_ceb
        pops = zip(df_pop_ceb['Total Population'], # Zip DataFrame columns of interest: pops
                    df_pop_ceb['Urban population (% of total)'])
        pops_list = list(pops)   # Turn zip object into list: pops_list
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list] # Use list comprehension to create new DataFrame column 'Total Urban Population'
        data = data.append(df_pop_ceb)         # Append DataFrame chunk to data: data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')  # Plot urban population data
    plt.show()
fn = 'ind_pop_data.csv' # Set the filename: fn
plot_pop('ind_pop_data.csv', 'CEB') # Call plot_pop for country code 'CEB'
plot_pop('ind_pop_data.csv', 'ARB') # Call plot_pop for country code 'ARB'
'''
print('importing files .txt CSV/....')
'''# Open a file: archivo_para_leer
archivo_para_leer = open('New_Text_Document.txt', 'r') #r of read, it could be also 'w' if we want to modify it
print(archivo_para_leer.read()) # Print it
print(archivo_para_leer.closed) # Check whether file is closed
archivo_para_leer.close() # Close file
print(archivo_para_leer.closed) # Check again whether file is closed

with open('New_Text_Document.txt') as archivo_para_leer2: #lee las 3 primeras lineas de new text document
    print(archivo_para_leer2.readline())
    print(archivo_para_leer2.readline())
    print(archivo_para_leer2.readline())
'''
print('Using NumPy to import flat files')
'''
# import numpy as np # Import package
digitos_txt = 'digits.txt' # Assign filename to variable: digitos_txt
digits = np.loadtxt(digitos_txt, delimiter=',') # Load file as array: digits
print(type(digits)) # Print datatype of digits ----> <class 'numpy.ndarray'>
im = digits[21, 1:] # Select and reshape a row
im_sq = np.reshape(im, (28, 28))
plt.imshow(im_sq, cmap='Greys', interpolation='nearest') # Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.show()
###There are a number of arguments that np.loadtxt() takes that you'll find useful:
###delimiter changes the delimiter that loadtxt() is expecting.
###You can use ',' for comma-delimited.
###You can use '\t' for tab-delimited.
###skiprows allows you to specify how many rows (not indices) you wish to skip
###usecols takes a list of the indices of the columns you wish to keep.
file = 'digits_header.txt' # Assign the filename: file
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2]) # Load the data: data the file you're importing is tab-delimited, you want to skip the first row and you only want to import the first and third columns.
print(data) # Print data
'''
print( 'Importing a csv')
'''
##import pandas as pd
nombredelCSV = 'ejemplo_del_curro.csv'
datitos = pd.read_csv(nombredelCSV)
print(datitos.head())
#convert de dataframe to a numpy array
datitos_array = datitos.values
'''
print('use pandas to import Excel spreadsheets and how to list the names of the sheets in any loaded .xlsx file')
'''# Import pandas import pandas as pd
filexls = 'battledeath.xlsx'  # Assign spreadsheet filename: file
xlssss = pd.ExcelFile(filexls) # Load spreadsheet: xls  ###Pass the correct argument to pd.ExcelFile() to load the file using pandas, assigning the result to the variable xls.
print(xlssss.sheet_names) # Print sheet names  ###you can retrieve a list of the sheet names using the attribute spreadsheet.sheet_names.

# Load a sheet into a DataFrame by name: df1
df1 = xlssss.parse('bdonly')
print(df1.head()) # Print the head of the DataFrame df1
df2 = xlssss.parse(0) # Load a sheet into a DataFrame by index: df2
print(df2.head()) # Print the head of the DataFrame df2
'''
print('Hello world SQL and first things with SQL')
'''
#Esto no funciona xk no tengo implementado  los packetes ni los SQLite Chinook y tal.
from sqlalchemy import create_engine # Import packages
import pandas as pd
engine = create_engine('sqlite:///Chinook.sqlite') # Create engine: engine
con = engine.connect() # Open engine connection: con
rs = con.execute("SELECT * from Album") # Perform query: rs
df = pd.DataFrame(rs.fetchall()) # Save results of the query to DataFrame: df
con.close() # Close connection
print(df.head()) # Print head of DataFrame df

#Now the same than above but including:
##Select specified columns from a table;
##Select a specified number of rows;
##Import column names from the database table.

# Open engine in context manager
with engine.connect() as con: # Perform query and save results to DataFrame: df
    rs = con.execute("SELECT LastName, Title FROM Employee") ##Execute the SQL query that selects the columns LastName and Title from the Employee table. Store the results in the variable rs.
    df = pd.DataFrame(rs.fetchmany(3)) #Apply the method fetchmany() to rs in order to retrieve 3 of the records. Store them in the DataFrame df.
    df.columns = rs.keys() #Using the rs object, set the DataFrame's column names to the corresponding names of the table columns.
print(len(df)) # Print the length of the DataFrame df
print(df.head()) # Print the head of the DataFrame df


#In this interactive exercise, you'll select all records of the Employee table for which 'EmployeeId' is greater than or equal to 6.
#Packages are already imported as follows:
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///Chinook.sqlite') # Create engine: engine
# Open engine in context manager
with engine.connect() as con:  # Perform query and save results to DataFrame: df
    rs = con.execute("SELECT * from Employee WHERE EmployeeId >=6")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
print(df.head()) # Print the head of the DataFrame df

##In this interactive exercise, you'll select all records of the Employee table and order them in increasing order by the column BirthDate.

engine = create_engine('sqlite:///Chinook.sqlite') # Create engine: engine
with engine.connect() as con: # Open engine in context manager
    rs = con.execute("SELECT * FROM Employee ORDER BY BirthDate")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()     # Set the DataFrame's column names
print(df.head()) # Print head of DataFrame
'''
print("SQL Querys with Pandas")
'''
# Import packages
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Chinook.sqlite') # Create engine: engine
df = pd.read_sql_query("SELECT * FROM Album", engine) # Execute query and store records in DataFrame: df
##The remainder of the code is included to confirm that the DataFrame created by this method is equal to that created by the previous method that you learned.
print(df.head()) # Print head of DataFrame
with engine.connect() as con: # Open engine in context manager and store query result in df1
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()
print(df.equals(df1)) # Confirm that both methods yield the same result



###You'll build a DataFrame that contains the rows of the Employee table for which the EmployeeId is greater than or equal to 6 and you'll order these entries by BirthDate.

# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query(
    "SELECT * FROM Employee WHERE EmployeeId >= 6 ORDER BY BirthDate",
    engine
)

# Print head of DataFrame
print(df.head())
'''
print('SQL with INNER JOIN')
'''
print("INNER JOIN The power of SQL lies in relationships between tables: INNER JOIN")
##Here, you'll perform your first INNER JOIN! You'll be working with your favourite SQLite database, Chinook.sqlite. For each record in the Album table, you'll extract the Title along with the Name of the Artist. The latter will come from the Artist table and so you will need to INNER JOIN these two tables on the ArtistID column of both.

# Open engine in context manager
with engine.connect() as con: # Perform query and save results to DataFrame: df
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID") ##Assign to rs the results from the following query: select all the records, extracting the Title of the record and Name of the artist of each record from the Album table and the Artist table, respectively. To do so, INNER JOIN these two tables on the ArtistID column of both.
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
print(df.head())  # Print head of DataFrame df

## ULTIMO EJEMPLO USANDO WHERE
# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000"  , engine)
# Print head of DataFrame
print(df.head())

'''
print("Exploring and cleaning CSVs data")
'''
# Import pandas
#import pandas as pd
# Read the file into a DataFrame: df
dfff = pd.read_csv('miejemplo.csv')

print(dfff.head(2)) # Print te muestra n las primeras (n=5 por default, se pueden cambiar) of dfff
print(dfff.tail()) # Print las 5 ultimas, (n se puede cambiar como con head)
print(dfff.shape) # Print the shape el numero de (raws, columnas)
print(dfff.columns) # Print the columns en una lista
print(dfff.info()) # Print the info of df

#perfom Frecuency count of a column
print(dfff.Edad.value_counts(dropna = False)) #  Ojo con caracteres especiales y espacios esto no anda.
#te dice
#el dropna te dice el numero valores que faltan, te los pone como NaN.
#value_counts te dice el n` de values que hay en la columna Edad o la que sea, en orden descendnete
#incluso si es un objeto te los cuenta como un int
print(dfff['Color'].value_counts(dropna = False) )# lo mismo que arriba pero escrito  de otra forma
print(dfff.Edad.value_counts(dropna = False) )#the number of observations for each Edad in our data
dfff.Edad.value_counts(dropna = False).head() #show only the top 5 Age
#We expect this column to be numeric and stored as string
print(dfff.describe()) #returns info only from  columns with numeric type. return the number of Non-missing values, mean, standar deviation std, min, max,....
'''
print('Visualizing with histograms')
'''import matplotlib.pyplot as plt
dfff['Edad'].plot(kind='hist', rot=70, logx=True, logy=True) #Histograma
plt.show()
dfff.boxplot(column='Edad', by='Color', rot=90) #te compara dos columnas
plt.show()
dfff.plot(kind='scatter', x='Edad', y='Altura', rot=70) #para comparar dos columnas con int
plt.show()
'''
print('Melting, pivoting y concatenando concat DATAFRAMES')
'''
edad_altura_melted = pd.melt(frame=dfff, id_vars=['Pais','Color']) ##The id_vars represent the columns of the data you do not want to melt (i.e., keep it in its current shape), while the value_vars represent the columns you do wish to melt into rows. By default, if no value_vars are provided, all columns not set in the id_vars will be melted.
#A a la variable y al Value se le puede cambiar el nombre: aqui por ejemoplo le ponemos measurement and reading:
#edad_altura_melted = pd.melt(frame=dfff, id_vars=['Pais','Color'], var_name='measurement', value_name='reading')
print(edad_altura_melted)

pivoteando = pd.read_csv('Para_pivotear.csv')
weather_tidy = pivoteando.pivot(index= 'date', columns='element',values='value')
print(weather_tidy)
#si hay values duplicados como en el ejemplo tmin sale dos veces tenemos que usar el pivot_table
#en .pivot_table() especificamos como pivotear los valores duplicados
pivoteando2 = pd.read_csv('Para_pivotear2.csv')
weather_tidy2 = pivoteando.pivot_table(values='value', index= 'date', columns='element', aggfunc = np.mean)
#el agg func le dice qu'e hacer con los valores duplicados, en este caso le decimos que haga la media con np.mean
print(weather_tidy2)
airquality_pivot_reset = weather_tidy2.reset_index()


#CONCATENAR DATAFRAMES
concatenando1 = pd.read_csv('para_concatenar1.csv')
concatenando2 = pd.read_csv('para_concatenar2.csv')
print(pd.concat([concatenando1,concatenando2], ignore_index=True)) #Te concatena dos dataframes pero si no le pones el ignore Index te los pone como uno debajo de otro, es decir habr'ra varios elementos (2 si hemos concatenado dos dataframes) con los mismos valores porque habr'a raws repetidos
'''
print('GLOB muchos files, luego se pueden concatenar o lo que sea')
'''import glob
files_con_nombre_parecido = glob.glob('nombre_parecido_*.csv') #te busca todos los files con lo que sea que haya en la estrellita
#se puede poner ? en vez de estrellita y te buscar'a los ?=enteros con ese nombre
print(files_con_nombre_parecido)
lista_de_CSVs_con_nombre_parecido=[]
for filename in files_con_nombre_parecido:
	data = pd.read_csv(filename)
	lista_de_CSVs_con_nombre_parecido.append(data)
pd.concat(lista_de_CSVs_con_nombre_parecido)
'''
print('MERGING DATA en tablas con datos sobre las mismas cosas pero con ordenes diferentes')
'''stat_populations = pd.read_csv('merging_data1.csv')
state_codes = pd.read_csv('merging_data2.csv')
print(pd.merge(left=stat_populations, right=state_codes, on=None, left_on='state', right_on='name'))
'''
print('Changing Data type .astype() y pd.to_numeric()')
'''
##ESTO SOLO LO HE COPIADO Y PEGADO, NO VA a funcionar
df['treatment b'] = df['treatment b'].astype(str) #cambia el tipo de Treatmen b a STR
df['sex'] = df['sex'].astype('category') ##cambia la columna sex a tipo 'category'
df.dtypes

df['treatment a'] = pd.to_numeric(df['treatment a'], errors='coerce') ## cambia Treatment a a numeric
#esto cambia el tipo de dato como lo de arriba a numerico y si hay algun dato mal que lo fuerza, lo coerce a que en vez de error nos ponga NaN.
'''
print('regular expresions to clean strings') #Libreria RE .match .search .compile   .findall .apply
'''
import re
#queremos poner numeros de telefono en el formato: xxx-xxx-xxxx y que nos diga si esta bien o mal
prog = re.compile('\d{3}-\d{3}-\d{4}')  ## de otro ejercicio '^[A-Za-z\.\s]*$'
result = prog.match('123-456-7890')
print(bool(result)) #ser'a true si el telefono est'a bien puesto en el formato americano
result2 = prog.match('1123-456-7890')
print(bool(result2)) #como el result2 '1123-456-7890' está mal puesto nos devuelve false
#\d* para cualqueir numero
#\$ para el dolar
#\. para la coma
#\d{2} para un numero en concreto
#^loquesea$ para que coga exactamente esos valores.
pattern = re.compile('\$\d*\.\d{2}')
result_ejercicillo = pattern.match('$17.89')
print(bool(result_ejercicillo))

matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana') #te da una lista de los numeros en un string
print(matches)

#A capital letter, followed by an arbitrary number of alphanumeric characters.
#Use [A-Z] to match any capital letter followed by \w* to match an arbitrary number of alphanumeric characters.
pattern3 = bool(re.match(pattern='\w*', string='Australia')) #TODO no lo entiendo :)
print(pattern3)

#ejemplo_para_apply = pd.read_csv('ejemplo_para_apply.csv')
#print(ejemplo_para_apply)
#print(ejemplo_para_apply.apply(np.mean, axis=1))


#import re
from numpy import NaN

pattern = re.compile('^\$\d*\.\d{2}$')
ejemplo_para_apply2= pd.read_csv('ejemplo_para_apply2.csv')

def diff_money(row, pattern):
    icost = row['Initial Cost']
    tef = row['Total Est. Fee']

    if bool(pattern.match(icost)) and bool(pattern.match(tef)):
        icost = icost.replace("$","")
        tef = tef.replace("$","")

        icost=float(icost)
        tef=float(tef)

        return icost - tef
    else:
        return (NaN)
ejemplo_para_apply2['diff'] = ejemplo_para_apply2.apply(diff_money, axis=1, pattern=pattern)
print(ejemplo_para_apply2)
### NO ME FUNCIONAAAAA Me devuelve una columna con to do NaN no s'e por que !
##TODO revisar lo de arriba y hacer una funcion para el curro que dandole una columna de EONs y SN me los limpie y me ponga (for Serial Number...)

## Ejemplo del curso, en una columna que pone Male y Female te crea otra columna y te pone 0 si es female y 1 si es male
'''
'''
# Define recode_gender()
def recode_gender(gender):
    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0

    # Return 1 if gender is 'Male'    
    elif gender == 'Male':
        return 1

    # Return np.nan    
    else:
        return np.nan


# Apply the function to the sex column
tips['recode'] = tips.sex.apply(recode_gender)

# Print the first five rows of tips
print(tips.head())
'''
print('Lambda function')
'''
def my_square(x):
    return x ** 2
df.apply(my_square) #df es un dataframe cualqueira
#El equivalente en lambda function ser'ia
df.apply(lambda x: x ** 2)
'''
'''# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())
'''
print('drop duplicates') #quitar duplicados de un dataframe
'''
df = df.drop_duplicates() #df es un dataframe cualqueira con missing values.
#esto te quita la row entera
print(df)
tips_dropped = ejemplodedafaframe.dropna() #te crea otro dataframe tips dropped sin
tips_dropped.info()
'''
print('fill missing values') #.fillna()
'''tips_nan['sex'] = tips_nan['sex'].fillna('missing') #esto mete 'missing' en todos los missing values de la columna SEX
tips_nan[['total_bill' , 'size']] = tips_nan[['total_bill', 'size']].fullna(0) #se puede hacer para varias columnas a la vez con doble [[]] y esa vez relleno los missing values con ceros 0

#podemos fill los missing values con la media en vez de ceros o "missing"
mean_value = tips_nan['tip'].mean()
print(mean_value)
tips_nan['tip']= tips_nan['tip'].fillna(mean_value)
tips_nan.info()
'''
print('assert')
'''assert 1 ==1 #esto no me devuelve nada porque es verdad porque es true
assert 1 ==2 #si le das algo falso te va a devolver un error
#.notnull() will return True if there is a value, false if there is a missing value
#.all() para mirar todos los valores del dataframe
assert google.Close.notnull().all() #google es un dataframe con missing values, Close no se, notnull y all estan arriba
#esto me devuelve errores porque es Falso porque hay missing values en el dataframe
google_0 = google.fillna(value = 0 ) #creamos otros dataframe que rellenamos los missing values con ceros 0
#y ahora el assert deber'ia no devolver nada,
assert google_0.Close.notnull().all() #esto no nos devuleve nada porque ya hemos rellenado todos los datos.

#Write an assert statement to confirm that there are no missing values in ebola:
assert ebola.notnull().all().all() # Assert that there are no missing values
#The first .all() method will return a True or False for each column, while the second .all() method will return a single True or False.
# se neceistan dos .all() porque es un dataframe o algo as'i no lo he entendido muy bien
# Assert that all values are >= 0
assert (ebola >= 0).all().all()
'''

print('Review of pandas DataFrames') #data frames son tablas con index en los raws y columnas. Las columnas son Panda Series
print("Building DataFrames")
'''
#import pandas as pd

df_ejemplo = pd.read_csv('miejemplo.csv')
print(df_ejemplo)

#####As an example, you can extract the rows that contain 'US' as the country of origin using df[df['origin'] == 'US'].
print(df_ejemplo[df_ejemplo['Nombre'] =='Fernando']) #me imprime todas las filas que contien Fernando en la columna Nombre

print(type(df_ejemplo))
print(df_ejemplo.shape) #me dice el n de rows y de columnas (rows, columns)
print(df_ejemplo.columns) #me da un pandas index con el nombre de las columnas
print(df_ejemplo.index) # me da los index de los rows
print(df_ejemplo.iloc[:3,:])#me corta y selecciona el dataframe desde la fila 0 a la 2, y todas las columnas
#:3 desde el principio hasta la raw 5 (non inclusive) as'i que me mostrara hasta las filas 0,1,2
#: me muestra todas las columnas
#[filas, columnas]
print(df_ejemplo.iloc[-3:,:]) # desde la 3 fila empezando por abajo hasta el final, mostrando todas las columnas
print(df_ejemplo.head()) # me muestra las 5 primeras. se puede poner .head(3) y me muestra las 3 primeras
print(df_ejemplo.tail()) # me muestra las 5 ultimas. se puede poner .head(3) y me muestra las 3 ultimas
print(df_ejemplo.info()) # the kind of index, column labels, n` of rows and columns y el data tipe de cada columna
df_ejemplo.iloc[::3,-1] = np.nan #podemos rellenar los huecos vacios con NaN
print(df_ejemplo.head()) #vemos que a los sitios vacios le hemos metido nan
#las columnas de los Dataframes son Panda Series
#A Panda Series is a 1dimension labelled NumPy array
#A DataFrame is a 2dimensions labelled array whose columns are Series
low = df_ejemplo['Edad']
print(type(low)) # me da el tipo de Panda series
print(low.head()) # me da las primeras filas de la columna1
#to extract numerical series entries from the Series use the values atribute
lows = low.values #esto me da un numpy array
print(type(lows))
print(df_ejemplo.describe()) #calcula la media, desviaci'on, percentiles y cosas de las columnas con valores numericos
print(df_ejemplo['Edad'].describe())

print(df_ejemplo['Edad'].count()) # returns the number of not Null entries for numerical cloumns
print(df_ejemplo.Edad.count()) # lo mismo escrito de otra forma
print(df_ejemplo[['Edad', 'Altura']].count())

print(df_ejemplo['Edad'].mean()) #aplica la media a la columna Edad of the not null things
print(df_ejemplo.mean()) #aplica la media a todos los valores numericos del dataframe of the not null things

print(df_ejemplo.std()) #standar diviation of the not null values
print(df_ejemplo.median()) #la media
cuantile = 0.5
print(df_ejemplo.quantile(cuantile)) #los cuantiles el 0.5 es la media
cuantiles = [0.25, 0.75]
print(df_ejemplo.quantile(cuantiles)) #mas cuantiles se les puede meter una serie
print(df_ejemplo.min()) #da el minimo y para los strings coge el 'ultimo por orden alfabetico
print(df_ejemplo.max())

print(df_ejemplo['Nombre'].unique()) #nombres unicos si hubiese

#filtering:
#
#indicesdelacolunna = df['columna'] =='un valor de la columna que no se repite' #crea
#df_con_Valores_de_la_columna = df.loc[indicesdelacolunna, :] # es una practica habitual filtrar y crear sub dataframes
'''




print("Create dataframe from Dictonary")
'''
ejemplo_dictionary = {'weekday': ['Sun', 'Sun', 'Mon','Mon'],
                               'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
                               'visitors': [139, 237 , 326, 456],
                                'signups': [7, 12 ,3, 5]}
df_ejemplo2_from_dictionary = pd.DataFrame(ejemplo_dictionary)
print(df_ejemplo2_from_dictionary) #notice that the INDEX are the row labels are the 0,1,2,3.... by default
print ('Create a DataFrame from lists')
weekdays = ['Sun', 'Sun', 'Mon','Mon']  #list1
cities = ['Austin', 'Dallas', 'Austin', 'Dallas'] #list 2
visitors = [139, 237 , 326, 456] #list3
signups = [7, 12 ,3, 5]   #list 4
list_labels = ['city', 'signups', 'visitors', 'weekday']  #contiene los column labels
list_cols = [cities, signups, visitors, weekdays]   #contiene los column entries for each column ITS A list of lists!
zippeando  = list(zip(list_labels, list_cols)) #zip of tuples
print(zippeando)
dictionary_from_lists = dict(zippeando) #hace un dictionary con dict de zippeando
df_ejemplo3_from_lists = pd.DataFrame(dictionary_from_lists)
print(df_ejemplo3_from_lists)
'''
print('Broadcasting un DataFrame') #añadir columna con cosass
'''df_ejemplo2_from_dictionary ['Nueva columna'] = 0 #esto crea una columna nueva con to do ceros 0
print(df_ejemplo2_from_dictionary)
'''
print('export csv to csv or excel')
'''df_ejemplo2_from_dictionary.to_csv('exportando_un_csv.csv')
df_ejemplo2_from_dictionary.to_csv('exportando_un_csv2_con_tab.csv', sep = '\t')
df_ejemplo2_from_dictionary.to_excel('exportando_un_csv_a_excel.xlsx')
'''
print('Broadcasting with dict')
'''#import pandas as pd
alturas_aleatorias = [59 , 123, 3,52, 35, 632,234, 123]
diccionario_ejemplonosecual = {'alturas': alturas_aleatorias, 'sex': 'M'}
df_ejemplo4_dictionary_broadcasting = pd.DataFrame(diccionario_ejemplonosecual)
print(df_ejemplo4_dictionary_broadcasting)
'''
print('Cambiar el columns y labels de un DataFrame')
'''df_ejemplo4_dictionary_broadcasting.columns = ['Alturass en CM', 'Sexo de ejemplo'] #podemos meter cualquier tipo de lista siempre que tenga la longitud del Dataframe
df_ejemplo4_dictionary_broadcasting.index = ['A','B','C','D','E','F','G','H']
print(df_ejemplo4_dictionary_broadcasting)
'''
#MAS COSAS SOBRE PANDAS DATAFRAMES

#no_funciona_ejemplo = pd.read_csv('Nombredel.csv', header = None, names = Listaejemplo, na_values = '-1' , parse_dates = ([0,1,2]) )
#header = None asume que la primera linea no son las columnas y mete 0 , 1 , 2 ,3 ... pero tambi'en podemos meterle un string con nombres
#names = lista mete los nombres de las culmnas, que con header = None eran 0,1,2,3... ahora son la lista Lista que queramos
#na_values = '-1' los valores que tengan -1 les va a meter NaN
#na_values = {'culumna' : ['-1', 'No aplicable', 'naaan' ]} podemos usar tambien una lista o un diccionario para los na_values por si hay muchas cosas para definir el NaN
#parse_dates = ([0,1,2]) las columnas 0 1 y 2 coneitnen los anios, meses y dias y si junto las columnas 0 1 y 2 puedo hacer un dates, con parse dates y se queda solo una columna, el nombre por defecto year_month_day se puede cambiar luego con un .index
'''# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))
plt.show()

# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', cumulative=True, normed=True, bins=30, range=(0,.3))
plt.show()'''
print('indexing times series pandas con Date  format yyyy-mm-dd hh:mm:ss') #tambien muchas cosas con .loc
'''
#import pandas as pd
indexeando_mal= pd.read_csv('indexing_datos_con_yyyymmdd.csv')
print(indexeando_mal)
indexeando = pd.read_csv('indexing_datos_con_yyyymmdd.csv', parse_dates=True, index_col= 'Date')
# juntamos las primeras columnas y las ponemos como index, ahora los index no son 0, 1 , 2... son las fechas y se llaman Date.
print(indexeando)
print(indexeando.loc['2015-02-02 21:00:00', 'Company'])
print(indexeando.loc['2015-02-02']) # puedes elegir un dia en particularr
#print(indexeando.loc['February 5,2015']) #no fundcionan
#print(indexeando.loc['2015-Feb-5']) ## no funcionan
print(indexeando.loc['2015-02']) # the whole month
print(indexeando.loc['2015']) # the whole year
print(indexeando.loc['2015-02-02':'2015-02-04']) #puedes seleccionar un rango separnado por :
'''
print('converting strings to datetime ')
'''
evening_2_02 = pd.to_datetime(['2015-02-02 08:30:00','2015-02-02 21:00:00', '2015-02-02 22:00:00']) # nete una columna nueva con la hora dada
print(evening_2_02)
print(indexeando.reindex(evening_2_02)) #rellena con  NaN los valores huecos
print(indexeando.reindex(evening_2_02, method = 'ffill')) #rellena con lo mismo que la columna de arriba los valores huecos
print(indexeando.reindex(evening_2_02, method = 'bfill')) #rellena con lo mismo que la columna de abajo los valores huecos
'''
print('resampling')
'''
media_por_dia = indexeando.resample('D').mean() #te hace la media por dias 'D' representa los dias pone NaN a los NaN, se puede cambiar con .ffill() por ejemplo
#la D se puede sustituir por W o otra cosa:
# min, t = minute, H= hour, D = Day, B = business day,  W = week, M = month, Q=quarter,  A = year
# Solo saca la media de las columnas numricals
print(media_por_dia)  #devuelve un DataFrame
total_por_dia = indexeando.resample('D').sum() # la suma por dia
print(total_por_dia)
el_max_de_total_por_dia = indexeando.resample('D').sum().max()
print(el_max_de_total_por_dia)
print(indexeando.resample('W').count())
#Downsampling:
    #print(indexeando.loc[:,'Units'].resample('2W').sum())   #el 2W es para 2 semanas de frecuencia
#upsimpling
#print(indexeando.loc['2015-2-2', '2015-2-5', 'Units']) no me anda
'''

print('reviewing more PandaFrames')
'''
#import pandas as pd

df_ejemplo = pd.read_csv('miejemplo.csv')
print(df_ejemplo)
print(df_ejemplo.shape) #te dice el numero de filas y columnas
print(df_ejemplo['Edad'][2]) #indexing [column][index]
print(df_ejemplo.Edad[2])
print(df_ejemplo.loc[2, 'Edad'])  #.loc[index, columna] con los nombres
print(df_ejemplo.iloc[2,1]) # lo mismo que el loc pero con los numeros en vez de los nombres
print(df_ejemplo[['Altura', 'Edad']]) #esto devuleve otro dafaframe, se puede poner en el orden que queramos
print('indexing Pandaframes')
print(df_ejemplo['Edad'][1:3])
print(df_ejemplo.loc[:, 'Edad':'Altura']) #all the raws from columns from Edad to Altura
print(df_ejemplo.loc[2:4, :]) #all columns in some raws
print(df_ejemplo.loc[2:4, 'Color':'Altura']) #some rows and some columns
print(df_ejemplo.iloc[2:4, 3:4]) # lo mismo que el de arriba pero con iLOc que hay que meterle numeros no los nombres de las columnas o los index
print(df_ejemplo.loc[2:4, ['Edad','Altura']]) # se pueden selecionar listas
print(df_ejemplo.iloc[[2,4], [3,4]]) # se pueden selecionar listas
print(df_ejemplo.loc[1:3:-1])
print(df_ejemplo.Edad>30) #devuelve TRUES OR FALSE
print(df_ejemplo[df_ejemplo.Edad>30])
mayores_de_30 = df_ejemplo.Edad>30
print(df_ejemplo[mayores_de_30])
print(df_ejemplo[(df_ejemplo.Edad>30) & (df_ejemplo.Altura<135)])
df_ejemplo2 = df_ejemplo.copy() #.copy() copia el dataframe a otro dataframe
df_ejemplo2['Columna Extra'] = [0, 0, 0,'cuatro','cinco','seis']

print(df_ejemplo2)
print(df_ejemplo2.loc[:, df_ejemplo2.all()]) #.all incluye todas las columnas que tienen todos los valores distintos a cero
#df_ejemplo2['Columna Extra'] = [0, 0, 0,0, 0, 0]
print(df_ejemplo2.loc[:, df_ejemplo2.any()]) #any non cero column entries, excluira a las columnas que son solo ceros, quita la columna 0 0 0  0 0 0
print(df_ejemplo.loc[:, df_ejemplo.isnull().any()]) # devuelve las columnas que tienen alg'un NaN
print(df_ejemplo2.loc[:, df_ejemplo2.isnull().any()])  # devuelve las columnas que tienen alg'un NaN
print(df_ejemplo.loc[:, df_ejemplo.notnull().all()]) # devuelve ALL the columns que no tienen NaN (notNull)
print(df_ejemplo.dropna(how='any')) #me quita las ROWS que tienen NaN by contracts how='all' will keep this rows
print(df_ejemplo.Edad[df_ejemplo.Altura>166])
'''
#TODO aprender mas sobre el metodo .apply, .map sobre datafranes
'''# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)

# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())'''

print('reviewing Pandas Data Structures')

#import pandas as pd
prices = [10.70, 10.86, 10.74, 10.71, 10.79]
shares = pd.Series(prices)
print(shares)
days = ['Mon' , 'Tue', 'Wed' , 'Thur', 'Fri']
shares = pd.Series(prices, index= days) #creamos un Panda series con index Days.
shares2 = shares #(para un ejemplo mas abajo)
print(shares)
print(shares.index.name) #printea el name attribute que si no le ponemos nada es 'None'
shares.index.name = 'weekday' #ahora el indice de la serie tendr'a como un titulillo al principio
print(shares) #ahora pondra weekday de titulillo  (weekday) al printear la serie
# si fuese un Dataframe tmb se puede poner titulo a las columnas shares.columns.name = 'PRODUCTS'
#NO PODEMOS MODIFICAR LOS INDEX UNO POR UNO: shares.index[2] = 'Wednesday' ----> dara error y shares.index[:4]= ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] ----> tmb dar'a error
#solo se puede cambiar all the index
shares.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] #pero esto borra el nombre que le hemos puesto antes
print(shares)
ejemplo_multiIndex = pd.read_csv('ejemplo_multiIndex.csv')
print(ejemplo_multiIndex)
ejemplo_multiIndex = ejemplo_multiIndex.set_index(['state', 'month'])
print(ejemplo_multiIndex)
print(ejemplo_multiIndex.index)
ejemplo_multiIndex = ejemplo_multiIndex.sort_index()
print(ejemplo_multiIndex)

print('pivoting a dataframe')
'''
ejemplo_pivotando = pd.read_csv('ejemplo_pivoting.csv')
print(ejemplo_pivotando)
ej_pivotado = ejemplo_pivotando.pivot(index ='weekday', columns = 'city', values ='visitors')
print(ej_pivotado)
signups_pivot = ejemplo_pivotando.pivot(index ='weekday', columns = 'city', values= 'signups')
print(signups_pivot)
pivot = ejemplo_pivotando.pivot(index ='weekday', columns = 'city')
print(pivot)
'''
print('tmb hay la funcion .stack y .unstack') #we are dealing with heralkical levles.
#.swaplevel tmb hace cosas con los dataframes.
print('tmb hay una funcion pd.melt')
#Recall from the video that the goal of melting is to restore a pivoted DataFrame to its original form, or to change it from a wide shape to a long shape. You can explicitly specify the columns that should remain in the reshaped DataFrame with id_vars, and list which columns to convert into values with value_vars. As Dhavide demonstrated, if you don't pass a name to the values in pd.melt(), you will lose the name of your variable. You can fix this by using the value_name keyword argument.


print('Categricals and Groupby Dataframes') #groupby es de pandas
'''
ventasej = pd.DataFrame(
    {
        'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
        'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
        'bread': [139, 237, 326, 456],
        'butter': [20, 45, 70, 98]
    }
)

print(ventasej)
#podríamos filtrar por booleanos para contar las ventas hechas en Sunday y contarlas
print(ventasej.loc[ventasej['weekday']== 'Sun'].count())
#Alternativamente podmeos groupby la columna weekday y contar las entradas por cada valor distinto encontrado
print(ventasej.groupby('weekday').count())
#GRoupby tmb funciona con mean(), std(),  sum(), first(), last(), max(),min()
print(ventasej.groupby('weekday')['bread'].sum())
print(ventasej.groupby('weekday')[['bread','butter']].sum())
#multilevel groupby
print(ventasej.groupby(['city','weekday']).mean())
#podemos groupby un panda series que tenga el mismo index que el dataframe
clientesej = pd.Series(['Dave', 'Alice', 'Bob', 'Alice'])
print(clientesej)
print(ventasej.groupby(clientesej)['bread'].sum())

print(ventasej['weekday'].unique()) # me dice cuantas veces aparece cada value
'''
print('grouby with Multiple aggregation .agg ') #.agg
'''print(ventasej.groupby('city')[['bread','butter']].max()) #esto es un single aggregation
print(ventasej.groupby('city')[['bread','butter']].agg(['max','sum']))
def data_range (series):
    return series.max()- series.min()
print(ventasej.groupby('city')[['bread','butter']].agg(data_range))  #agg acepta funciones
#.agg tambi'en acepta dictionaries
print(ventasej.groupby(clientesej)[['bread','butter']].agg({'bread':'sum','butter': data_range}))
'''
print('idxmax(): returns  Row or column label where maximum value is located') #dataframe.idxmax()
print('idxmin():  returns Row or column label where minimum value is located')

print('reviewing Dataframes again')
#filenames = ['sales-jan-2015.csv', 'sales-feb-2015.csv'] # creamos una lista con los nombre de los files names
#dataframes = [] #otra lista vaicia de dataframes
#for f in filenames:
#    dataframes.append(pd.read_csv(f))

#tambien lo podemos hacer con un list comprenhension
#dataframes = [pd.read_csv(f) for f in filenames]  #cargando tambi'en la liste filenames del ejmplo de justo encima
print('importamos muchos CSVs usando la  glob')
#from glob import glob
# filenames = glob('sales*.csv') #la estrellita puede ser cualquier numero del 0 al 9
# dataframes = [pd.read_csv(f) for f in filenames]

#reindex
#ordered = ['Jan', 'Apr', 'Jul', 'Oct'] #creamos una lista con los nuevos indices
#w_mean2 = w_mean.reindex(ordered) #con la funcion reindex creamos un nuevo dataframe con el indice de la lista anterior
#print(w_mean2)

#tambien se puede usar sort_index
#w_mean2.sort_index() #esto har'ia parceido a lo de antes, pero esto reordena los indices que ya tiene el pandafarme.
#w_mean.reindex(w_max.index) #tmb puedes coger el indice de otro dataframe y ponerselo al que quieres

#w_mean3 = w_mean.reindex(['Jan', 'Apr', 'Dec']) #si reindexamos con los indices que ya tiene Jan y Apr y le metemos uno que no tiene Dec pondr'a NAN en Dec

#w_max.reindex(w_mean3.index) #podmemos poner el indice de otro dataframe pa saber si los dos tienen los mismos indices, si no lo tiene entonces pondra NaN
#podemos borrar los rows que no tienen valores con .dropna()
#w_max.reindex(w_mean3.index).dropna() #esto borra las filas que no tiene nadatos

print('merge and merge_ordered on Dataframes from pd ')
#pd.merge()   pd.merge_ordered(), the pd.merge_asof()