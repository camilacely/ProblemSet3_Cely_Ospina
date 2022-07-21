###################################
## Big Data - Problem Set 3 #######
# Maria Camila Cely , Sara Ospina #
###### Julio 2022 #################
###################################

#"Prediction of housing values"

#####################
# 1. Data Acquisition
#####################

## clean environment
rm(list=ls())

## Llamar/instalar las librerias

require(pacman)
p_load(tidyverse,    #Para limpiar los datos
       caret,        #Para la clasificación y regresiones
       rio,          #Para importar datos
       modelsummary, # msummary
       gamlr,        
       class,
       ggplot2,
       skimr,
       rvest,
       dplyr,
       stargazer,
       gtsummary,
       expss,
       fastAdaboost,
       randomForest,
       xgboost,
       glmnet,
       pROC,
       class,
       sf,
       leaflet,
       tmaptools,
       osmdata, 
       skim) #por ahora llame todas las del problem set 2

predict<- stats::predict  #con esto soluciono el problema de que haya mas de una libreria con este comando


#####################
##cargar los datos #
#####################

##Establecer el directorio

#setwd
#setwd("C:/Users/SARA/Documents/ESPECIALIZACIÓN/BIG DATA/GITHUB/ProblemSet3_Cely_Ospina")
setwd("C:/Users/Camila Cely/Documents/GitHub/ProblemSet3_Cely_Ospina")

#traer las bases de train y de test

test<-readRDS("stores/test.Rds")     #11.150 obs
train<-readRDS("stores/train.Rds")  #107.567 obs

#####################
##1. Limpieza de datos: 
###################

###### LO PRIMERO
#cuales son las variables que aparecen tanto en train como en test

intersect(names(test), names(train))

# [1] "property_id"     "ad_type"         "start_date"      "end_date"        "created_on"      "lat"            
#[7] "lon"             "l1"              "l2"              "l3"              "rooms"           "bedrooms"       
#[13] "bathrooms"       "surface_total"   "surface_covered" "currency"        "title"           "description"    
#[19] "property_type"   "operation_type" 

#vamos a ver en general como se comportan las variables

#fechas
summary(train$start_date) #estas fechas si dan bien
summary(train$end_date) #estas fechas, por encima de la mediana, tienen cosas raras, ejemplo año 4000 o 9000
#las fechas se refieren al inicio (y fin, cuando hay) del anuncio en Properati, no consideramos que sean utiles para la prediccion
summary(train$created_on) #estas dan bien

#coordenadas
var_lab(train$lat) = "Latitud"
var_lab(train$lon) = "Longitud"

#localizacion general
var_lab(train$l2) = "Departamento"
var_lab(train$l3) = "Municipio"

#cuartos
summary(train$rooms) #esta tiene 53.606 NAs, promedio= 2.98
summary(train$bedrooms) #sin NAs, promedio =3.08 #propongo usar esta porque tienen casi la misma informacion, mismo min y max

var_lab(train$bedrooms) = "Num de cuartos"

#baños
summary(train$bathrooms) # tiene 30.074 NAs

#   Min. 1st Qu.  Median   Mean   3rd Qu.  Max.    NA's 
#  1.000   2.000   2.000   2.723   3.000  20.000   30074 
#tiene este valor maximo de 20 bathrooms que creo que es outlier, podemos eliminarlo

#que hacer con los NAs?
#opcion 1= imputar que las viviendas tienen minimo un baño
#opcion 2= eliminar estas observaciones

#Por ahora creo que lo mejor es eliminar las observaciones porque, si bien las viviendas deben tener minimo un baño #REVISAR
#podrian tener dos e imputarles uno solo nos puede estar modificando mucho las predicciones
#intuicion = el numero de baños afecta mucho el precio de la vivienda

#train <- train %>% drop_na(c("bathrooms")) #aqui quedaria una base con 77.493 obs
#POR AHORA NO LA VOY A CORRER PORQUE NO ESTOY SEGURA DE LA DECISION DE ELIMINAR LOS NAs
#SEGUN LO QUE VIMOS EL MARTES CON EDUARD, LO MEJOR ES IMPUTARLO CON INFORMACION DE LOS VECINOS

var_lab(train$bathrooms) = "Num de banos"

#area
summary(train$surface_total) #NAs = 49.936 #estan saliendo demasiados NAs, no podemos eliminar todo esto o nos quedamos sin obs
#va a tocar imputarle valores promedio de acuerdo con otras caracteristicas

summary(train$surface_covered) #57.515 NAs

#    Min. 1st Qu.  Median   Mean  3rd Qu.  Max.    NA's  #surface_total
#    11      70     108     173     189   108800   49936 

#   Min. 1st Qu.  Median   Mean  3rd Qu.   Max.    NA's   #surface_covered
#   1.0    73.0   110.0   148.3   188.0  11680.0   57515   #notar que no se comportan tan tan diferente

#por algun motivo no hay datos demasiado buenos de area, revisar

#podriamos crear una variable que relacione estas dos

#train <- train %>% 
#  mutate(area_descubierta = (train$surface_total - train$surface_covered  ))

#summary(train$area_descubierta) 
#head(train$area_descubierta)  #la verdad se generan 96.625 NAs y valores negativos #las voy a dejar para que no corran porque por ahora creo que esta variable no es necesaria

#creo que lo mejor sera guiarnos por surface_total

var_lab(train$surface_total) = "Area total"


#dinero
summary(train$currency) #aqui nos dice que estan medidas en COP
head(train$currency)
sum(train$currency == "USD") #sale 0  #vemos que no hay valores en dolares
sum(train$currency == "COP") #107567  #la totalidad de los precios estan en COP


#anuncio
head(train$title) #aqui sale el titulo de la oferta, ejemplo= "hermosa casa en venta"
var_lab(train$title) = "Titulo anuncio"

head(train$description) #aqui dice los detalles, ejemplo = barrio, "muy iluminado", "moderno", etc7
var_lab(train$description) = "Descripcion anuncio"

#tipo de propiedad
head(train$property_type) #esta nos dice si es casa o apartamento
var_lab(train$property_type) = "Tipo de propiedad"

#venta o arriendo
head(train$operation_type) #aqui nos dice si es venta o arriendo
sum(train$operation_type == "Venta") #107.567 #la totalidad de propiedades estan en venta

#Entonces = las variables que considero que mas nos importan son las siguientes

#train$start_date
#train$lat
#train$lon
#train$l2
#train$l3
#train$bedrooms
#train$bathrooms
#train$surface_total
#train$title
#train$description
#train$property_type

#VOLVER LAS VARIABLES QUE SE NECESITEN COMO AS.FACTOR

train$l2 <- as.factor(train$l2)       
class(train$l2)

train$l3 <- as.factor(train$l3)       
class(train$l3)

train$property_type <- as.factor(train$property_type)       
class(train$property_type)


###### LO SEGUNDO
#vamos a ver como se comporta price
#recordar que price esta en train pero no en test

summary(train$price)
hist(train$price) #hay demasiados precios muy altos que arrastran todo
#en el histograma vemos que la mayoria de precios estan hacia los 500 millones de pesos

#hagamos una variable que este en millones de pesos para que sea mas facil de interpretar

train <- train %>% 
  mutate(precio_millones = (train$price / 1000000 ))

as.numeric (train$precio_millones)

hist(train$precio_millones)


#aqui voy a hacer una tabla que nos diga como se comportan las variables hasta el momento

train %>%
  select(price, start_date, l2, bedrooms, bathrooms, surface_total, property_type) %>%
  tbl_summary()

#de aqui obtenemos a grandes rasgos lo siguiente=

# PRECIO PROMEDIO = 460 millones
# DEPARTAMENTOS = 80% estan en Cundinamarca y 20% en Antioquia
# CUARTOS = En promedio hay 3
# BANOS = En promedio hay 2 (pero hay muchos missings)
# AREA TOTAL = Promedio 108 m2 (pero hay muchos missings) #79.845
# TIPO VIVIENDA = 76% apartamento, 24% casa


##################################################
#PRIMER INTENTO DE COMPLETAR LAS VARIABLES DE AREA
###################################################


#Antes de eliminar los missings vamos a ver si podemos "completar" un poco la variable de area con lo que haya en 
#surface_total y en surface_covered

summary(train$surface_covered) #87.368 missings
summary(train$surface_total) #79.845 missings

#voy a duplicar ambas variables para poderlas trabajar

train <- train %>% 
  mutate(area_total = train$surface_total)

train <- train %>% 
  mutate(area_cubierta = train$surface_covered)

#quiero que ambas sean numeric

as.numeric (train$area_total)
as.numeric (train$area_cubierta)

class (train$area_total) #esta sale labelled numeric
class (train$area_cubierta)

var_lab(train$area_total) = NULL #le quito el label para evitar este problema
as.numeric (train$area_total)
class (train$area_total) #corregido

#ahora voy a crear una variable que "sume" esas dos

train <- train %>% 
  mutate(area = if_else( is.na(area_total)==TRUE , train$area_cubierta, train$area_total)) #le pido que mantenga area_total lo mas que pueda a menos que sea NA, entonces que ponga area_cubierta

#habiendo hecho eso, las variables con solo NAs en "area" son las que realmente son missings >> las que tengan NA, solo esas voy a imputar
table(is.na(train$area))  #70.588 missings 

#notar que redujimos un poco porque originalmente teniamos #87.368 missings y #79.845 missings respectivamente

#EN TODO CASO, ESTA PENDIENTE IMPUTARLE VALORES A ESAS 70MIL OBSERVACIONES
#lo haremos de dos maneras= sacando el dato de la descripcion del anuncio
# o= imputando por valores de k-nearest neighbors


######################################################################################
######################################################################################
######################################################################################

#PRIMERO: CREAR LOS POLIGONOS DE CHAPINERO Y DE EL POBLADO

#CHAPINERO

upla<-read_sf("stores/upla/UPla.shp") #totalidad de la ciudad

up_chapinero <- upla %>% filter(UPlNombre
                                %in%c("EL REFUGIO", "PARDO RUBIO", "CHICO LAGO", "CHAPINERO")) #aqui agarro las upz de chapinero (localidad)

up_chapinero <- st_transform(up_chapinero, 4326)

chapinero <- getbb(place_name = "UPZ Chapinero, Bogota", 
                   featuretype = "boundary:administrative", 
                   format_out = "sf_polygon") %>% .$multipolygon #aca se pone lo de multipolygon exactamente porque tiene un pedazo "suelto" (San Luis)

leaflet() %>% addTiles() %>% addPolygons(data=chapinero) #la diferencia es que esta incluye la UPZ San Luis #sale de OSM
leaflet() %>% addTiles() %>% addPolygons(data=up_chapinero) #esta solo incluye las 4 UPZ antes de subir a la calera #sale de Ignacio


#EL POBLADO

poblado <- getbb(place_name = "Comuna 14 - El Poblado", 
                   featuretype = "boundary:administrative", 
                   format_out = "sf_polygon") 

leaflet() %>% addTiles() %>% addPolygons(data=poblado)


#SEGUNDO
#Ahora, vamos a realizar el corte de las observaciones de train
#como ya tenemos chapinero y poblado creadas, hay que volver train "espacial" en solo esas areas

#ESPACIALIZAR TRAIN
train_f <- st_as_sf(x=train,coords=c("lon","lat"),crs=4326)
leaflet() %>% addTiles() %>% addCircleMarkers(data=train_f, col="red") #salen las obs de todo el pais

#CHAPINERO

#version con san luis
train_chap <- st_crop (train_f, chapinero)
leaflet() %>% addTiles() %>% addPolygons(data=chapinero) %>% addCircleMarkers(data=train_chap, col="red") #funciona

#version sin san luis
train_upchap <- st_crop (train_f, up_chapinero)
leaflet() %>% addTiles() %>% addPolygons(data=up_chapinero) %>% addCircleMarkers(data=train_upchap, col="red") #funciona tambien
#LA VERDAD PROPONGO GUIARNOS POR UP_CHAPINERO PORQUE SAN LUIS ES UN BARRIO MUY DISTINTO AL RESTO DE LA LOCALIDAD Y PODRIA ESTARNOS ARRASTRANDO CON INFO QUE NO ES


#POBLADO
train_pobl <- st_crop (train_f , poblado)
leaflet() %>% addTiles() %>% addPolygons(data=poblado) %>% addCircleMarkers(data=train_pobl, col="red") #SALE AL REVES, PENDIENTE AYUDA POR SLACK


train_pobla <- train_f[poblado,]

leaflet() %>% addTiles() %>% addPolygons(data=poblado) %>% addCircleMarkers(data=train_pobla, col="red") #SALE AL REVES, PENDIENTE AYUDA POR SLACK


#voy a intentar hacerlo a partir de un XML que saque de medellin.gov.co

install.packages("XML")

library("XML")
library("methods")

xml_medellin <- xmlParse(file = "stores/metadata_medellin.xml")

print(xml_medellin)

######################################################################################
######################################################################################
######################################################################################
                 
#MANZANAS BOGOTA

mbog <- read_sf ("stores/manzanasbogota/MGN_URB_MANZANA.shp")
st_crs(mbog)<-4326  

mchap <- st_crop (mbog, up_chapinero)

leaflet() %>% addTiles() %>% addPolygons(data=mchap , color="red") %>% addCircles(data=train_upchap)


#MANZANAS MEDELLIN

mmed <- read_sf ("stores/manzanasantioquia/MGN_URB_MANZANA.shp")
st_crs(mmed)<-4326  

sf_use_s2(FALSE)
mpobl <- st_crop (mmed, poblado)

#leaflet() %>% addTiles() %>% addPolygons(data=mmed , color="red") #no funciona precisamente porque el poligono poblado no esta cortando bien
#PENDIENTE ARREGLAR


######################################################################################
######################################################################################
######################################################################################

###PREDICTORS COMING FROM EXTERNAL SOURCES

#Ahora si vamos a ver como se comporta, por ejemplo, estar cerca a bares, restaurantes, etc

#CHAPINERO

opq(bbox = getbb ("UPZ Chapinero, Bogota")) #para solo chapinero

available_features()

available_tags("amenity") 
#aqui voy a poner los que creo que pueden ser relevantes

#"bar"
#"brothel"
#"bus_station"
#"cafe"
#"cinema"
#"gambling"
#"gym"
#"hospital"
#"love_hotel"
#"marketplace"
#"public_building"
#"restaurant"
#"school"
#"theatre"
#"university"

#BAR
bar_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="bar")

class(bar_chap)

bar_chap <- bar_chap %>% osmdata_sf() 
bar_chap

bar_chap <- bar_chap$osm_points %>% select(osm_id,amenity)
bar_chap

leaflet() %>% addTiles() %>% addCircleMarkers(data=bar_chap , color="red")
#intuicion= buena variable porque esta muy localizada en ciertas zonas 

#BROTHEL
brothel_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="brothel")

brothel_chap <- brothel_chap %>% osmdata_sf() 

brothel_chap <- brothel_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=brothel_chap , color="red") 
#intuicion= me sale UNO pero veremos si es significativo


#BUS STATION
bus_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="bus_station")

bus_chap <- bus_chap %>% osmdata_sf() 

bus_chap <- bus_chap$osm_points %>% select(osm_id,amenity)


leaflet() %>% addTiles() %>% addCircleMarkers(data=bus_chap , color="red")
#intuicion= creo que va a resultar significativo porque esta muy marcado por eje vial de la caracas


#CAFE
cafe_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="cafe")

cafe_chap <- cafe_chap %>% osmdata_sf() 

cafe_chap <- cafe_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=cafe_chap , color="red")
#intucion= este NO creo que sirva porque esta demasiado bien distribuido en todo el barrio, casi no nos dice nada

#CINEMA
cine_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="cinema")

cine_chap <- cine_chap %>% osmdata_sf() 

cine_chap <- cine_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=cine_chap , color="red")
#intuicion= puede ser buena porque hay pocos en el barrio y puede dar cuenta de centros comerciales y de areas de ocio


#GAMBLING= 0 points
#GYM= 0 points

#HOSPITAL
hosp_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="hospital")

hosp_chap <- hosp_chap %>% osmdata_sf() 

hosp_chap <- hosp_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=hosp_chap , color="red")
#intuicion= puede estar dando significativo porque se agrupan mucho hacia el sur de la localidad y hacia la caracas


#LOVE_HOTEL = 0 points (lastima porque creo que era buena variable)
#MARKETPLACE = 0 points
#PUBLIC BUILDING = 4 points pero no me quiere dejar sacarlos

#RESTAURANT
rest_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="restaurant")

rest_chap <- rest_chap %>% osmdata_sf() 

rest_chap <- rest_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=rest_chap , color="red")
#intuicion= pasa lo mismo que con cafe, hay por todas partes en este barrio, 
#por eso NO creo que nos ayude a predecir bien los precios
#PERO la voy a dejar porque igual si uno hace zoom si se agrupa en ciertos corredores


#SCHOOL
sch_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="school")

sch_chap <- sch_chap %>% osmdata_sf() 

sch_chap <- sch_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=sch_chap , color="red")
#intuicion= ambivalente, creo que si puede servir pero lo malo es que tiene mezclados colegios de distintas "gamas" entonces se pueden confundir las senales


#THEATRE
th_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="theatre")

th_chap <- th_chap %>% osmdata_sf() 

th_chap <- th_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=th_chap , color="red")
#intuicion= no hay muchos, creo que podria estar mandando alguna senal

#UNIVERSITY
uni_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="university")

uni_chap <- uni_chap %>% osmdata_sf() 

uni_chap <- uni_chap$osm_points %>% select(osm_id,amenity)

leaflet() %>% addTiles() %>% addCircleMarkers(data=uni_chap , color="red")
#intuicion= debe ser buena variable porque hay muchas pero estan MUY concentradas en ciertos corredores

#CONCLUSION
#MEJORES VARIABLES QUE VAMOS A UTILIZAR

#"bar"
#"brothel"
#"bus_station"
#"cinema"
#"hospital"
#"restaurant"
#"school"
#"theatre"
#"university"

#LAS METEREMOS TODAS Y QUE RANDOM FORESTS DECIDA EL RESTO

#eso de arriba eran amenities, ahora quiero ver parques que hacen parte de "leisure"

#PARK
park_chap <- opq(bbox = st_bbox(up_chapinero)) %>% 
  add_osm_feature(key = "leisure", value = "park") %>%
  osmdata_sf() %>% .$osm_points %>% select(osm_id,name)

leaflet() %>% addTiles() %>% addCircleMarkers(data=park_chap , color="red")
#intuicion= en esta pagina tienen reportados como parques cuadraditos de pasto que yo creo que van a sobreestimar el efecto
#en todo caso me parece bien dejarla por ahora

#para controlar lo de parques muy pequenos, voy a sacar una de playground, que debe haber menos

#PLAYGROUND
play_chap <- opq(bbox = st_bbox(up_chapinero)) %>% 
  add_osm_feature(key = "leisure", value = "playground") %>%
  osmdata_sf() %>% .$osm_points %>% select(osm_id,name)

leaflet() %>% addTiles() %>% addCircleMarkers(data=play_chap , color="red")
#intuicion= mucho mejor, ahora si nos salen menos parques, mucho mas realista, vemos que la mayoria estan en chapinero alto y mas hacia el norte

######################################################################################
#HABIENDO CREADO LAS VARIABLES DE INTERES, AHORA LO QUE HARE ES MEDIR LAS DISTANCIAS A ESOS PUNTOS

#prueba

train_f$dist_play <- st_distance(x=train_f, y=play_chap)

head(train_f$dist_play) ##VOY ACA 



######################################################################################
######################################################################################
######################################################################################


###PREDICTORS COMING FROM DESCRIPTION ## aqui empezar a hacer lo que vimos en la clase con Eduard (martes) - usar base completa (train)


##############################
#######Imputar valores########
##############################

##en train 

#Le indico cual es mi df y la latitud u el codigo que voy a usar >> no entiendo por que queda con una variable menos #RTA: PORQUE JUNTA LAT Y LON EN UNA SOLA VARIABLE LLAMADA GEOMETRY
train_f <- st_as_sf(x=train,coords=c("lon","lat"),crs=4326)

train_f <- train_f %>%
  mutate (titlemin = str_to_lower(string = train_f$title))

train_f <- train_f %>%
  mutate (descriptionmin = str_to_lower(string = train_f$description))


#####AREA########
#Revisar los missing values de las areas
table(is.na(train_f$area)) #tiene 70588 NA

#Patrones para imputar metraje
x1 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+m2" ## pattern
x2 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mts"
x2 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mts2"
x3 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+metros"
x4 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+m2"
x5 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mt2"
x6 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+m²"

x7 = "[:space:]+[:digit:]+[:space:]+m2"
x8 = "[:space:]+[:digit:]+[:space:]+mts"
x9 = "[:space:]+[:digit:]+[:space:]+mts2"
x10 = "[:space:]+[:digit:]+[:space:]+metros"
x11 = "[:space:]+[:digit:]+[:space:]+m2"
x12 = "[:space:]+[:digit:]+[:space:]+mt2"
x13 = "[:space:]+[:digit:]+[:space:]+m²"

x14 = "[:space:]+[:digit:]+m2"
x15 = "[:space:]+[:digit:]+mts"
x16 = "[:space:]+[:digit:]+mts2"
x17 = "[:space:]+[:digit:]+metros"
x18 = "[:space:]+[:digit:]+m2"
x19 = "[:space:]+[:digit:]+mt2"
x20 = "[:space:]+[:digit:]+m²"

#imputamos los valores de area que estan NA con los patrones 
train_f = train_f %>% 
  mutate(area = ifelse(is.na(area)==T,
                       str_extract(string=train_f$description , pattern= 
                       paste0(x1,"|",x2,"|",x3,"|",x4,"|",x5,"|",x6,"|",x7,"|",x8,"|",x9,"|",x10,"|",
                              x11,"|",x12,"|",x13,"|",x14,"|",x15,"|",x16,"|",x17,"|",x18,"|",x19,"|",x20)),
                       area))

#verificamos como cambio NA
table(is.na(train_f$area)) #tiene 48580 NA, imputamos 22008

table(train_f$area)
sum(table(train_f$area))

view(train_f$area)

str_extract (string = train_f$area, ) ##pendiente terminar



#pendiente:

#cambiar en la variable train_f$area todas las comas por puntos
#despues, extraer solo los numeros


#####BANOS########
table(is.na(train_f$bathrooms)) #30074 NA

#patrones para banos
y1 = "[:space:]+[:digit:]+[:space:]+ba"
y2 = "[:space:]+[:digit:]+ba"

#imputar los NA de baños con los patrones
train_f = train_f %>% 
  mutate(bathrooms = ifelse(is.na(bathrooms)==T,
                       str_extract(string=train_f$description , pattern= 
                       paste0(y1,"|",y2)),
                       bathrooms))
table(is.na(train_f$bathrooms)) #20934 NA, imputamos 9140. las observaciones imputadas quedan con el ba, no se si eso afecte 

######NIVEL##### 
z1 = "[:space:]+[:alpha:]+[:space:]+piso"
z2 = "piso+[:space:]+[:alpha:]+[:space:]"
z3 = "[:space:]+[:digit:]+[:alpha:]+piso"
w1 = "penthouse"
w2 = "pent-house"
w3 = "pent house"

#creamos una nueva variable >> no estoy segura, algunos si los toma bien pero tambien toma cosas como el tipo de piso o cosas que escriben despues que no tienen que ver 
train_f = train_f %>% 
  mutate(nivel = str_extract(string=train_f$description , pattern= paste0(z1,"|",z2,"|",z3,"|",w1,"|",w2,"|",w3)))
table(train_f$nivel)
table(is.na(train_f$nivel)) #75611 NA, no se si valga la pena

####Estrato##### 
v1 = "estrato+[:space:]+[:digit:]+[:space:]"
v2 = "estrato+[:digit:]+[:space:]"

#creamos nueva variable
train_f = train_f %>% 
  mutate(estrato = str_extract(string=train_f$description , pattern= paste0(v1,"|",v2)))
table(train_f$estrato)
table(is.na(train_f$estrato)) #106210 NA >> creo que esta es mejor sacarla del DANE


####info DANE#### >>> tendriamos que traer manzanas y no entiendo como se limpia el archivo de manzanas 

#por manzana calculo la mediana del numero de cuartos, nu de personas y el estrato
## load data
mnz_censo = import("http://eduard-martinez.github.io/data/fill-gis-vars/mnz_censo.rds")

## about data
browseURL("https://eduard-martinez.github.io/teaching/meca-4107/6-censo.txt")

## spatial join
house_censo = st_join(house_mnz,mnz_censo)
colnames(house_censo)










table(train_f$title)

ch = "chapinero"## pattern
p = "el poblado"
p2= "poblado"









