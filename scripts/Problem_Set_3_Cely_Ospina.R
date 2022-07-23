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
setwd("C:/Users/SARA/Documents/ESPECIALIZACIÓN/BIG DATA/GITHUB/ProblemSet3_Cely_Ospina")
#setwd("C:/Users/Camila Cely/Documents/GitHub/ProblemSet3_Cely_Ospina")

#traer las bases de train y de test

test<-readRDS("stores/test.Rds")     #11.150 obs
train<-readRDS("stores/train.Rds")  #107.567 obs

#####################
##0. Union de bases: 
###################

#este es un paso recomendado para transformar ambas bases de una vez y no tener que repetir todos los comandos en train y en test

#creamos una dummy llamada test que toma valor de 1 para las observaciones originales de test y 0 para las de train
test$test <- 1 
train$test <- 0 

test$price <- NA #Se crea la variable precio en Test como NAs para incluirla y poder combinar las bases, porque recordemos que en Test originalmente no viene Precio

train_test <- rbind(train, test) #train_f viene junta con test

##IMPORTANTE: CAMBIAR EL CODIGO PARA QUE LAS TRANSFORMACIONES LAS REALICE EN ESTA BASE COMPLETA


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


###########EL ANALISIS (EJ. SUMMARY) LO VOY A MANTENER SOLO EN TRAIN, PERO LOS LABELS POR EJEMPLO SI LOS VOY A CORRER EN LA BASE UNIFICADA
#vamos a ver en general como se comportan las variables 

#fechas
summary(train$start_date) #estas fechas si dan bien
summary(train$end_date) #estas fechas, por encima de la mediana, tienen cosas raras, ejemplo año 4000 o 9000
#las fechas se refieren al inicio (y fin, cuando hay) del anuncio en Properati, no consideramos que sean utiles para la prediccion
summary(train$created_on) #estas dan bien

#coordenadas
var_lab(train_test$lat) = "Latitud"
var_lab(train_test$lon) = "Longitud"

#localizacion general
var_lab(train_test$l2) = "Departamento"
var_lab(train_test$l3) = "Municipio"

#cuartos
summary(train$rooms) #esta tiene 53.606 NAs, promedio= 2.98
summary(train$bedrooms) #sin NAs, promedio =3.08 #propongo usar esta porque tienen casi la misma informacion, mismo min y max

var_lab(train_test$bedrooms) = "Num de cuartos"

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

var_lab(train_test$bathrooms) = "Num de banos"

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

var_lab(train_test$surface_total) = "Area total"


#dinero
summary(train$currency) #aqui nos dice que estan medidas en COP
head(train$currency)
sum(train$currency == "USD") #sale 0  #vemos que no hay valores en dolares
sum(train$currency == "COP") #107567  #la totalidad de los precios estan en COP


#anuncio
head(train$title) #aqui sale el titulo de la oferta, ejemplo= "hermosa casa en venta"
var_lab(train_test$title) = "Titulo anuncio"

head(train$description) #aqui dice los detalles, ejemplo = barrio, "muy iluminado", "moderno", etc7
var_lab(train_test$description) = "Descripcion anuncio"

#tipo de propiedad
head(train$property_type) #esta nos dice si es casa o apartamento
var_lab(train_test$property_type) = "Tipo de propiedad"

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

train_test$l2 <- as.factor(train_test$l2)       
class(train_test$l2)

train_test$l3 <- as.factor(train_test$l3)       
class(train_test$l3)

train_test$property_type <- as.factor(train_test$property_type)       
class(train_test$property_type)


###### LO SEGUNDO
#vamos a ver como se comporta price
#recordar que price esta en train pero no en test

summary(train$price)
hist(train$price) #hay demasiados precios muy altos que arrastran todo
#en el histograma vemos que la mayoria de precios estan hacia los 500 millones de pesos

#hagamos una variable que este en millones de pesos para que sea mas facil de interpretar

train <- train %>% 
  mutate(precio_millones = (train$price / 1000000 )) #no estoy segura de si crearla en train_test porque yo solo la hice para entender mejor, no creo que la vayamos a usar en las predicciones

as.numeric (train$precio_millones)

hist(train$precio_millones)


#aqui voy a hacer una tabla que nos diga como se comportan las variables hasta el momento

train %>%
  select(price, start_date, l2, bedrooms, bathrooms, surface_total, property_type) %>%
  tbl_summary() #solo en train porque es solo para entender

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

summary(train_test$surface_covered) #87.368 missings ### 97.371 en la base unificada
summary(train_test$surface_total) #79.845 missings   ### 88.936 en la base unificada

#voy a duplicar ambas variables para poderlas trabajar

train_test <- train_test %>% 
  mutate(area_total = train_test$surface_total)

train_test <- train_test %>% 
  mutate(area_cubierta = train_test$surface_covered)

#quiero que ambas sean numeric

as.numeric (train_test$area_total)
as.numeric (train_test$area_cubierta)

class (train_test$area_total) #esta sale labelled numeric
class (train_test$area_cubierta)

var_lab(train_test$area_total) = NULL #le quito el label para evitar este problema
as.numeric (train_test$area_total)
class (train_test$area_total) #corregido

#ahora voy a crear una variable que "sume" esas dos

train_test <- train_test %>% 
  mutate(area = if_else( is.na(area_total)==TRUE , train_test$area_cubierta, train_test$area_total)) #le pido que mantenga area_total lo mas que pueda a menos que sea NA, entonces que ponga area_cubierta

#habiendo hecho eso, las variables con solo NAs en "area" son las que realmente son missings >> las que tengan NA, solo esas voy a imputar
table(is.na(train_test$area))  #70.588 missings  ### 79.364 en la base unificada

#notar que redujimos un poco porque originalmente teniamos mas missings

#EN TODO CASO, ESTA PENDIENTE IMPUTARLE VALORES A ESAS 79MIL OBSERVACIONES
#lo haremos de tres maneras= sacando el dato de la descripcion del anuncio
# o= imputando por valores del promedio de las manzanas vecinas
# o= imputando valores del censo


######################################################################################
######################################################################################
######################################################################################

#PRIMERO: CREAR LOS POLIGONOS DE CHAPINERO Y DE EL POBLADO

#CHAPINERO

upla<-read_sf("stores/upla/UPla.shp") #totalidad de la ciudad

up_chapinero <- upla %>% filter(UPlNombre
                                %in%c("EL REFUGIO", "PARDO RUBIO", "CHICO LAGO", "CHAPINERO")) #aqui agarro las upz de chapinero (localidad) pero no agarro San Luis 

up_chapinero <- st_transform(up_chapinero, 4326)

chapinero <- getbb(place_name = "UPZ Chapinero, Bogota", 
                   featuretype = "boundary:administrative", 
                   format_out = "sf_polygon") %>% .$multipolygon #aca se pone lo de multipolygon exactamente porque tiene un pedazo "suelto" (San Luis)

leaflet() %>% addTiles() %>% addPolygons(data=chapinero) #la diferencia es que esta incluye la UPZ San Luis #sale de OSM
leaflet() %>% addTiles() %>% addPolygons(data=up_chapinero) #esta solo incluye las 4 UPZ antes de subir a la calera #sale de datos que nos proporciono Ignacio en la clase de spatial data


#EL POBLADO

poblado <- getbb(place_name = "Comuna 14 - El Poblado", 
                   featuretype = "boundary:administrative", 
                   format_out = "sf_polygon") 

leaflet() %>% addTiles() %>% addPolygons(data=poblado)


#SEGUNDO
#Ahora, vamos a realizar el corte de las observaciones de nuestra base
#como ya tenemos chapinero y poblado creadas, hay que volver las observaciones de caracter "espacial" en solo esas areas

#ESPACIALIZAR BASE
tt_sf <- st_as_sf(x=train_test,coords=c("lon","lat"),crs=4326)
leaflet() %>% addTiles() %>% addCircleMarkers(data=tt_sf, col="red") #salen las obs de todo el pais # tt_sf


#CHAPINERO
#LA VERDAD PROPONGO GUIARNOS POR UP_CHAPINERO PORQUE SAN LUIS ES UN BARRIO MUY DISTINTO AL RESTO DE LA LOCALIDAD Y PODRIA ESTARNOS ARRASTRANDO CON INFO QUE NO ES
#Y PRECISAMENTE ESTAMOS HACIENDO EL SUBSET POR BARRIO PARA HACER UNA "CAJITA" DE OBSERVACIONES PARECIDAS EN LAS QUE PODAMOS COMPARAR MEJOR

#version con san luis
#train_chap <- st_crop (train_f, chapinero)
#leaflet() %>% addTiles() %>% addPolygons(data=chapinero) %>% addCircleMarkers(data=train_chap, col="red") #funciona pero la voy a dejar para que no corra porque incluye San Luis

#version sin san luis
tt_upchap <- st_crop (tt_sf, up_chapinero)
leaflet() %>% addTiles() %>% addPolygons(data=up_chapinero) %>% addCircleMarkers(data=tt_upchap, col="red") #funciona tambien


#POBLADO
tt_pobl <- st_crop (tt_sf , poblado)
leaflet() %>% addTiles() %>% addPolygons(data=poblado) %>% addCircleMarkers(data=tt_pobl, col="red") #esta linea estuvo dando problemas, por seguridad voy a guardar una copia del objeto

#save(tt_pobl, file = "stores/tt_pobl.Rdata")
#save.image()

#tt_pobl<-readRDS("stores/tt_pobl.Rds")   #esto lo dejo para que no corra pero en caso de que vuelva a fallar el comando tenemos este backup


######################################################################################
######################################################################################
######################################################################################
                 
#MANZANAS BOGOTA

mbog <- read_sf ("stores/manzanasbogota/MGN_URB_MANZANA.shp")
st_crs(mbog)<-4326  

mchap <- st_crop (mbog, up_chapinero)

leaflet() %>% addTiles() %>% addPolygons(data=mchap , color="red") %>% addCircles(data=tt_upchap)


#MANZANAS MEDELLIN

mmede <- read_sf ("stores/manzanasantioquia/MGN_URB_MANZANA.shp")
st_crs(mmede)<-4326  

sf_use_s2(FALSE)
mpobla <- st_crop (mmede, poblado)                                                        

plot(mpobla)

leaflet() %>% addTiles() %>% addPolygons(data=mpobla , color="red") %>% addCircles(data=tt_pobl)

#como este mpobla tambien estuvo dando problemas, tambien le voy a guardar una copia de seguridad

#save(mpobla, file = "stores/mpobla.Rdata")
#save.image()

#mpobla<-readRDS("stores/mpobla.Rds") #solo correr esta linea en caso de que vuelva a fallar el comando


######################################################################################
######################################################################################
######################################################################################

###PREDICTORS COMING FROM EXTERNAL SOURCES

#Ahora si vamos a ver como se comporta, por ejemplo, estar cerca a bares, restaurantes, etc

###########
#CHAPINERO
##########

#available_features()
#available_tags("amenity") #estas dos lineas nos permiten inspeccionar que encontraremos en osm, las pongo para que no corran porque se demoran demasiado, pero de aqui es que saque los amenities que considere relevantes=

#amenities que creo que pueden ser relevantes==

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

#todas estas variables estan sacadas de la manera larga porque lo fui haciendo paso a paso como lo aprendimos con eduard, mas adelante en leisure aprendi una manera de hacerlo con menos lineas de codigo
#BAR
bar_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="bar")
bar_chap <- bar_chap %>% osmdata_sf() 
bar_chap <- bar_chap$osm_points %>% select(osm_id,amenity)
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
bus_chap <- bus_chap$osm_points %>% select(osm_id)
leaflet() %>% addTiles() %>% addCircleMarkers(data=bus_chap , color="red")
#intuicion= creo que va a resultar significativo porque esta muy marcado por eje vial de la caracas


#CAFE
cafe_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="amenity", value="cafe")
cafe_chap <- cafe_chap %>% osmdata_sf() 
cafe_chap <- cafe_chap$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cafe_chap , color="red")
#intucion= este NO creo que sirva mucho porque esta demasiado bien distribuido en todo el barrio, casi no nos dice nada... sin embargo a mayor zoom se ve que si se concentra en algunas esquinas...

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
uni_chap <- uni_chap$osm_points %>% select(osm_id)
leaflet() %>% addTiles() %>% addCircleMarkers(data=uni_chap , color="red")
#intuicion= debe ser buena variable porque hay muchas pero estan MUY concentradas en ciertos corredores

########################################
#CONCLUSION
#MEJORES VARIABLES QUE VAMOS A UTILIZAR

#"bar"
#"brothel"
#"bus_station"
#"cafe"
#"cinema"
#"hospital"
#"restaurant"
#"school"
#"theatre"
#"university"

#DE ESTAS LAS METEREMOS TODAS Y QUE RANDOM FORESTS DECIDA EL RESTO

#^^^^^^^^^^^^^
#eso de arriba eran amenities, ahora quiero ver parques que hacen parte de "leisure"


#PARK
park_chap <- opq(bbox = st_bbox(up_chapinero)) %>% 
  add_osm_feature(key = "leisure", value = "park") %>%
  osmdata_sf() %>% .$osm_points %>% select(osm_id,leisure) #esta es la manera corta de correr lo mismo

leaflet() %>% addTiles() %>% addCircleMarkers(data=park_chap , color="red")
#intuicion= en esta pagina tienen reportados como parques cuadraditos de pasto que yo creo que van a sobreestimar el efecto
#en todo caso me parece bien dejarla por ahora

#para controlar lo de parques muy pequenos, voy a sacar una de playground, que debe haber menos

#PLAYGROUND
play_chap <- opq(bbox = st_bbox(up_chapinero)) %>% 
  add_osm_feature(key = "leisure", value = "playground") %>%
  osmdata_sf() %>% .$osm_points %>% select(osm_id,leisure)

leaflet() %>% addTiles() %>% addCircleMarkers(data=play_chap , color="red")
#intuicion= mucho mejor, ahora si nos salen menos parques, mucho mas realista, vemos que la mayoria estan en chapinero alto y mas hacia el norte

#ambas variables nos sirven

#"park"
#"playground"

#ademas voy a meter una de grandes vias

#HIGHWAY
hw_chap <- opq(bbox = st_bbox(up_chapinero)) %>%
  add_osm_feature(key="highway", value="trunk")
hw_chap <- hw_chap %>% osmdata_sf() 
hw_chap <- hw_chap$osm_points %>% select(osm_id)
leaflet() %>% addTiles() %>% addCircleMarkers(data=hw_chap , color="red")


#CBD POINTS
centro_inter <- geocode_OSM("Torre Colpatria, Bogotá", as.sf = T) 
centro_inter <- centro_inter$point  %>% select(query)
leaflet() %>% addTiles() %>% addCircleMarkers(data=centro_inter , color="red")

av_chile <- geocode_OSM("Torre Banco de Occidente, Bogotá", as.sf = T) 
av_chile <- av_chile$point  %>% select(query)
leaflet() %>% addTiles() %>% addCircleMarkers(data=av_chile , color="red")

calle_100 <- geocode_OSM("World Trade Center, Bogotá", as.sf = T) 
calle_100 <- calle_100$point  %>% select(query)
leaflet() %>% addTiles() %>% addCircleMarkers(data=calle_100 , color="red")

usaquen <- geocode_OSM("Hacienda Santa Bárbara, Bogotá", as.sf = T) 
usaquen <- usaquen$point  %>% select(query)
leaflet() %>% addTiles() %>% addCircleMarkers(data=usaquen , color="red")

cbd_bog<- rbind(centro_inter , av_chile, calle_100, usaquen)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cbd_bog , color="red")


##################################################################################
#ahora conociendo las variables que nos sirven, voy a hacer lo mismo para medellin
##################################################################################


###########
#POBLADO
##########

#"bar"
#"brothel"
#"bus_station"
#"cafe"
#"cinema"
#"hospital"
#"restaurant"
#"school"
#"theatre"
#"university"

#BAR
bar_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="bar")
bar_pobl <- bar_pobl %>% osmdata_sf() 
bar_pobl <- bar_pobl$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=bar_pobl , color="red")

#BROTHEL
brothel_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="brothel")
brothel_pobl <- brothel_pobl %>% osmdata_sf() #### salen 0 points y en chapinero sale solo uno, creo que vamos a tener que no incluir esta variable

#BUS STATION
bus_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="bus_station")
bus_pobl <- bus_pobl %>% osmdata_sf() 
bus_pobl <- bus_pobl$osm_points %>% select(osm_id)
leaflet() %>% addTiles() %>% addCircleMarkers(data=bus_pobl , color="red") #salen muy poquitos, voy a probar con subway

#SUBWAY
sub_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="route", value="subway")
sub_pobl <- sub_pobl %>% osmdata_sf() 
sub_pobl <- sub_pobl$osm_points %>% select(osm_id)
leaflet() %>% addTiles() %>% addCircleMarkers(data=sub_pobl , color="red") #aqui si salen bastantes, la cosa es que me devuelve todos los de medellin y no todos los de poblado solamente...

#CAFE
cafe_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="cafe")
cafe_pobl <- cafe_pobl %>% osmdata_sf() 
cafe_pobl <- cafe_pobl$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cafe_pobl , color="red") #vemos que en chapinero los cafes estan mejor distribuidos, aqui estan super concentrados, buenisima variable 

#CINEMA
cine_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="cinema")
cine_pobl <- cine_pobl %>% osmdata_sf() 
cine_pobl <- cine_pobl$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cine_pobl , color="red") 

#HOSPITAL
hosp_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="hospital")
hosp_pobl <- hosp_pobl %>% osmdata_sf() 
hosp_pobl <- hosp_pobl$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=hosp_pobl , color="red") #parecido a como pasa en chapinero, se concentran en el eje fuerte de transporte publico

#RESTAURANT
rest_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="restaurant")
rest_pobl <- rest_pobl %>% osmdata_sf() 
rest_pobl <- rest_pobl$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=rest_pobl , color="red") #altamente concentrados tambien

#SCHOOL
sch_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="school")
sch_pobl <- sch_pobl %>% osmdata_sf() 
sch_pobl <- sch_pobl$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=sch_pobl , color="red") #altamente concentrados tambien

#THEATRE
th_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="theatre")
th_pobl <- th_pobl %>% osmdata_sf() 
th_pobl <- th_pobl$osm_points %>% select(osm_id,amenity)
leaflet() %>% addTiles() %>% addCircleMarkers(data=th_pobl , color="red")

#UNIVERSITY
uni_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="amenity", value="university")
uni_pobl <- uni_pobl %>% osmdata_sf() 
uni_pobl <- uni_pobl$osm_points %>% select(osm_id)
leaflet() %>% addTiles() %>% addCircleMarkers(data=uni_pobl , color="red")

#PARK
park_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="leisure", value="park")
park_pobl <- park_pobl %>% osmdata_sf() 
park_pobl <- park_pobl$osm_points %>% select(osm_id,leisure)
leaflet() %>% addTiles() %>% addCircleMarkers(data=park_pobl , color="red")

#PLAYGROUND
play_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="leisure", value="playground")
play_pobl <- play_pobl %>% osmdata_sf() 
play_pobl <- play_pobl$osm_points %>% select(osm_id,leisure)
leaflet() %>% addTiles() %>% addCircleMarkers(data=play_pobl , color="red")

#HIGHWAY
hw_pobl <- opq(bbox = st_bbox(poblado)) %>%
  add_osm_feature(key="highway", value="trunk")
hw_pobl <- hw_pobl %>% osmdata_sf() 
hw_pobl <- hw_pobl$osm_points %>% select(osm_id)
leaflet() %>% addTiles() %>% addCircleMarkers(data=hw_pobl , color="red") #nota= highway captura, para ambas ciudades, informacion muy parecida a la de transporte masivo

#CBD MEDELLIN
cbd_med <- geocode_OSM("Avenida el Poblado, Medellín", as.sf = T) 
cbd_med <- cbd_med$point  %>% select(query)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cbd_med , color="red")


######################################################################################
#QUIERO UNIR LAS VARIABLES DE UN MISMO TEMA DE AMBAS CIUDADES PORQUE LOS DATOS LOS TENEMOS COMO TOTALIDAD DE LAS DOS CIUDADES ENTONCES NO QUIERO SUMAR "POR APARTE" LAS DISTANCIAS

#"bar"
#"brothel" ##esta se cancela porque no obtuvimos suficientes observaciones
#"bus_station"
#"cafe"
#"cinema"
#"hospital"
#"restaurant"
#"school"
#"theatre"
#"university"
#"park"
#"playgound"
#"highway" (trunk)

bar <- rbind(bar_chap, bar_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=bar , color="red") #perfecto

bus <- rbind(bus_chap, bus_pobl, sub_pobl) #le sume metro a esta variable
leaflet() %>% addTiles() %>% addCircleMarkers(data=bus , color="red")

cafe <- rbind(cafe_chap, cafe_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cafe , color="red")

cinema <- rbind(cine_chap, cine_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cinema , color="red")

hospital <- rbind(hosp_chap, hosp_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=hospital , color="red")

restaurant <- rbind(rest_chap, rest_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=restaurant , color="red")

school <- rbind(sch_chap, sch_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=school , color="red")

theatre <- rbind(th_chap, th_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=theatre , color="red")

university <- rbind(uni_chap, uni_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=university , color="red")

park <- rbind(park_chap, park_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=park , color="red")

playground <- rbind(play_chap, play_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=playground , color="red")

highway <- rbind(hw_chap, hw_pobl)
leaflet() %>% addTiles() %>% addCircleMarkers(data=highway , color="red")

cbd <- rbind (cbd_bog, cbd_med)
leaflet() %>% addTiles() %>% addCircleMarkers(data=cbd , color="red")


######################################################################################
#AHORA HAY QUE UNIR LOS SUBSETS DE OBSERVACIONES PARA QUE QUEDE UNA UNICA BASE "SUBSETEADA"

tt_barrios <- rbind(tt_upchap, tt_pobl) #27.871 observaciones (la original tiene 118.717) (o sea estamos trabajando aqui con 1/4 de las observaciones aproximadamente)
leaflet() %>% addTiles() %>% addCircleMarkers(data=tt_barrios , color="red")


######################################################################################
# AHORA LO QUE HARE ES MEDIR LAS DISTANCIAS A ESOS PUNTOS DE INTERES COMO BARES ETC


#BAR
dist_bar <- st_distance(x=tt_barrios, y=bar) #matriz de n*j donde n es el numero de propiedades (tt_barrios) y j es el numero de amenities (bares) #ESTA MATRIZ ES ENORME Y LA VERDAD NO LA NECESITAMOS COMPLETA

#entonces lo que se hace es buscar por ejemplo el bar mas cercano para cada una de las propiedades
min_dist_bar <- apply(dist_bar,1,min)

#y posteriormente este es el valor que imputamos a la base tt_barrios
tt_barrios$dist_bar <- min_dist_bar

head(tt_barrios$dist_bar) #ya vemos que no sale a modo de matriz sino a modo de valor unico

#ahora lo hare con el resto

#BUS
dist_bus <- st_distance(x=tt_barrios, y=bus)
min_dist_bus <- apply(dist_bus,1,min)
tt_barrios$dist_bus <- min_dist_bus
head(tt_barrios$dist_bus)

#CAFE
dist_cafe <- st_distance(x=tt_barrios, y=cafe)
min_dist_cafe <- apply(dist_cafe,1,min)
tt_barrios$dist_cafe <- min_dist_cafe
head(tt_barrios$dist_cafe)

#CINEMA
dist_cinema <- st_distance(x=tt_barrios, y=cinema)
min_dist_cinema <- apply(dist_cinema,1,min)
tt_barrios$dist_cinema <- min_dist_cinema
head(tt_barrios$dist_cinema)

#HOSPITAL
dist_hospital <- st_distance(x=tt_barrios, y=hospital)
min_dist_hospital <- apply(dist_hospital,1,min)
tt_barrios$dist_hospital <- min_dist_hospital
head(tt_barrios$dist_hospital)

#RESTAURANT
dist_restaurant <- st_distance(x=tt_barrios, y=restaurant)
min_dist_restaurant <- apply(dist_restaurant,1,min)
tt_barrios$dist_restaurant <- min_dist_restaurant
head(tt_barrios$dist_restaurant)

#SCHOOL
dist_school <- st_distance(x=tt_barrios, y=school)
min_dist_school <- apply(dist_school,1,min)
tt_barrios$dist_school <- min_dist_school
head(tt_barrios$dist_school)

#THEATRE
dist_theatre <- st_distance(x=tt_barrios, y=theatre)
min_dist_theatre <- apply(dist_theatre,1,min)
tt_barrios$dist_theatre <- min_dist_theatre
head(tt_barrios$dist_theatre)

#UNIVERSITY
dist_university <- st_distance(x=tt_barrios, y=university)
min_dist_university <- apply(dist_university,1,min)
tt_barrios$dist_university <- min_dist_university
head(tt_barrios$dist_university)

#PARK
dist_park <- st_distance(x=tt_barrios, y=park)
min_dist_park <- apply(dist_park,1,min)
tt_barrios$dist_park <- min_dist_park
head(tt_barrios$dist_park)

#PLAYGROUND
dist_playground <- st_distance(x=tt_barrios, y=playground)
min_dist_playground <- apply(dist_playground,1,min)
tt_barrios$dist_playground <- min_dist_playground
head(tt_barrios$dist_playground)

#HIGHWAY
dist_highway <- st_distance(x=tt_barrios, y=highway)
min_dist_highway <- apply(dist_highway,1,min)
tt_barrios$dist_highway <- min_dist_highway
head(tt_barrios$dist_highway)

#CBD
dist_cbd <- st_distance(x=tt_barrios, y=cbd)
min_dist_cbd <- apply(dist_cbd,1,min)
tt_barrios$dist_cbd <- min_dist_cbd
head(tt_barrios$dist_cbd)

###NOTAR ALGO IMPORTANTE: LA DISTANCIA A BARES, ETC. SOLO ESTA MEDIDA CON RESPECTO A LAS OBSERVACIONES DE LOS BARRIOS ESPECIFICAMENTE (TT_BARRIOS)
###ESTO ES PORQUE SACAR LA DISTANCIA A TODOS LOS BARES DE LAS CIUDADES Y MEDIRLAS CON RESPECTO A TODAS LAS OBSERVACIONES ES COMPUTACIONALMENTE INMENSO #matriz n*j demasiado grande
###POR ESO SE REALIZA SOLO SOBRE EL SUBSET

###########################################################
#VOY A PONER UNA VARIABLE ESPACIAL RELACIONADA CON EL ORIENTE

#Intuicion: en nuestros barrios seleccionados y segun nuestro conocimiento, las propiedades tienen mayores precios mientras esten mas cerca de las montañas
# esto puede estar asociado a mas altos valores paisajisticos DIFICILES DE CAPTURAR PORQUE NO TENEMOS VARIABLES DE ALTURA DE SUELO
# a manera de coincidencia, en ambos barrios (chapinero y poblado) las montañas quedan al ORIENTE
#entonces creare una variable asociada con la variable de LONGITUD

#teniendo en cuenta que la longitud 0 es en Londres, y Colombia queda "a la izquierda" de Londres 
#nuestra longitud es NEGATIVA
#por lo tanto entre MAS CERCA A CERO este el valor, mas al ORIENTE está

#o sea que al crear una variable de ORIENTE, esta va a tener una correlacion con precio
#porque entre MAYOR sea la LONGITUD, MAYOR ES EL PRECIO (segun nuestra intuicion)

#ahora bien, lo malo es que eliminamos la variable de LONGITUD porque esta se fusionó con latitud para crear geometry
#entonces toca volverla a meter
#estaba en train_test

#recupero la variable de longitud asociada a property id (proviene de train_test)
tt_lon <- select(filter(train_test),c(property_id, lon)) 

#le llevo esta informacion a tt_barrios
tt_barrios <- left_join(tt_barrios, tt_lon)

#creo la variable de oriente
tt_barrios$oriente <- tt_barrios$lon
as.numeric (tt_barrios$oriente)

summary(tt_barrios$oriente)  

#importante= esta variable solo la podemos usar con el subset de los barrios
#si la corremos con las observaciones de la ciudad completa mandaria una mala señal porque por ejemplo medellin es un valle
#por fuera del poblado hay montañas en otros puntos cardinales
#lo mismo en bogota por ejemplo en colina campestre




##CON ESTO YA TENEMOS LAS VARIABLES ESPACIALES, PROCEDEMOS ENTONCES CON LAS VARIABLES DE TEXTO =

######################################################################################
######################################################################################
######################################################################################


###PREDICTORS COMING FROM DESCRIPTION 


##############################
#######Imputar valores########
##############################

##en train 

#Le indico cual es mi df y la latitud u el codigo que voy a usar >> no entiendo por que queda con una variable menos #RTA: PORQUE JUNTA LAT Y LON EN UNA SOLA VARIABLE LLAMADA GEOMETRY


tt_sf <- tt_sf %>%
  mutate (titlemin = str_to_lower(string = tt_sf$title))

tt_sf <- tt_sf %>%
  mutate (descriptionmin = str_to_lower(string = tt_sf$description))


#####AREA########
#Revisar los missing values de las areas
table(is.na(tt_sf$area)) #tiene 79364 NA

#Patrones para imputar metraje
x1 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+m2" ## pattern
x2 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mt"
x3 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mts"
x4 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mts2"
x5 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mts 2"
x6 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+metros"
x7 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+m2"
x8 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+mt2"
x9 = "[:space:]+[:digit:]+[:punct:]+[:digit:]+[:space:]+m+²"

x10 = "[:space:]+[:digit:]+[:space:]+m2"
x11= "[:space:]+[:digit:]+[:space:]+mts"
x12 = "[:space:]+[:digit:]+[:space:]+mts2"
x13 = "[:space:]+[:digit:]+[:space:]+mts 2"
x14 = "[:space:]+[:digit:]+[:space:]+metros"
x15 = "[:space:]+[:digit:]+[:space:]+m2"
x16 = "[:space:]+[:digit:]+[:space:]+mt2"
x17 = "[:space:]+[:digit:]+[:space:]+mt"
x18 = "[:space:]+[:digit:]+[:space:]+m+²"

x19 = "[:space:]+[:digit:]+m2"
x20 = "[:space:]+[:digit:]+mts"
x21 = "[:space:]+[:digit:]+mts2"
x22 = "[:space:]+[:digit:]+mts 2"
x23 = "[:space:]+[:digit:]+metros"
x24 = "[:space:]+[:digit:]+m2"
x25 = "[:space:]+[:digit:]+mt2"
x26 = "[:space:]+[:digit:]+mt 2"
x27 = "[:space:]+[:digit:]+m+²"



#imputamos los valores de area que estan NA con los patrones 
tt_sf = tt_sf %>% 
  mutate(area = ifelse(is.na(area)==T,
                       str_extract(string=tt_sf$description , pattern= 
                       paste0(x1,"|",x2,"|",x3,"|",x4,"|",x5,"|",x6,"|",x7,"|",x8,"|",x9,"|",x10,"|",
                              x11,"|",x12,"|",x13,"|",x14,"|",x15,"|",x16,"|",x17,"|",x18,"|",x19,"|",
                              x20,"|",x21,"|",x22,"|",x23,"|",x24,"|",x25,"|",x26,"|",x27)),
                       area))

#verificamos como cambio NA
table(is.na(tt_sf$area)) #tiene 53468 NA, imputamos 25896

table(tt_sf$area)
sum(table(tt_sf$area))

view(tt_sf$area)

str_extract (string = tt_sf$area, ) ##pendiente terminar



#pendiente:

#cambiar en la variable tt_sf$area todas las comas por puntos
#despues, extraer solo los numeros


#####BANOS########
table(is.na(tt_sf$bathrooms)) #34343 NA

#patrones para banos
y1 = "[:space:]+[:digit:]+[:space:]+baños"
y2 = "[:space:]+[:digit:]+[:space:]+banos"
y3 = "[:space:]+[:digit:]+[:space:]+baos"
y4 = "[:space:]+[:digit:]+baños"
y5 = "[:space:]+[:digit:]+banos"
y6 = "[:space:]+con baño+[:space:]"
y7 = "[:space:]+un baño+[:space:]"
y8 = "[:space:]+dos baños+[:space:]"
y9 = "[:space:]+dos baños+[:punct:]"
y10= "[:space:]+tres baños+[:space:]"
y11= "[:space:]+tres baños+[:punct:]"
y12= "[:space:]+cuatro baños+[:space:]"
y13= "[:space:]+cuatro baños+[:punct:]"



#imputar los NA de baños con los patrones
tt_sf = tt_sf %>% 
  mutate(bathrooms = ifelse(is.na(bathrooms)==T,
                       str_extract(string=tt_sf$description , pattern= 
                       paste0(y1,"|",y2,"|",y3,"|",y4,"|",y5,"|",y6,"|",y7,"|",y8,"|",y9,"|",y10,"|",y11,"|",y12,"|",y13)),
                       bathrooms))
table(is.na(tt_sf$bathrooms)) #18824 NA, imputamos 15519. las observaciones imputadas quedan con el ba, no se si eso afecte 

######NIVEL##### >> aqui no estoy segura, en medellin los ed pueden tener hasta 30pisos
w1 = "[:space:]+piso+[:space:]+[:digit:]"
w2 = "piso+[:space:]+[:digit:]"
w3 = "[:space:]+piso+[:space:]+[:digit:]+[:punct]"
w4 = "[:space:]+primer piso+[:space:]"
w5 = "[:space:]+segundo piso+[:space:]"
w6 = "[:space:]+tercer piso+[:space:]"
w7 = "[:space:]+cuarto piso+[:space:]"
w8 = "[:space:]+quinto piso+[:space:]"
w9 = "[:space:]+sexto piso+[:space:]"
w10 = "[:space:]+septimo piso+[:space:]"
w11 = "[:space:]+octavo piso+[:space:]"
w12 = "[:space:]+noveno piso+[:space:]"
w13 = "[:space:]+decimo piso+[:space:]"
w14 = "[:space:]+primer piso+[:punct:]"
w15 = "[:space:]+segundo piso+[:punct:]"
w16 = "[:space:]+tercer piso+[:punct:]"
w17 = "[:space:]+cuarto piso+[:punct:]"
w18 = "[:space:]+quinto piso+[:punct:]"
w19 = "[:space:]+sexto piso+[:punct:]"
w20 = "[:space:]+septimo piso+[:punct:]"
w21 = "[:space:]+octavo piso+[:punct:]"
w22 = "[:space:]+noveno piso+[:punct:]"
w23 = "[:space:]+decimo piso+[:punct:]"


#creamos una nueva variable >> no estoy segura, algunos si los toma bien pero tambien toma cosas como el tipo de piso o cosas que escriben despues que no tienen que ver 
tt_sf = tt_sf %>% 
  mutate(nivel = str_extract(string=tt_sf$description , pattern= paste0(w1,"|",w2,"|",w3,"|",w4,"|",w5,"|",w6,"|",w7,"|",w8,"|",w9,"|",w10,"|",w11,"|",w12,"|",
                                                                          w13,"|",w14,"|",w15,"|",w16,"|",w17,"|",w18,"|",w19,"|",w20,"|",w21,"|",w22,"|",w23)))
table(tt_sf$nivel)
table(is.na(tt_sf$nivel)) #101126 NA, no se si valga la pena ##de acuerdo

#####balcon/terraza/bbq####
v1 = "[:space:]+terraza+[:space:]"
v2 = "[:space:]+tiene terraza+[:space:]"
v3 = "[:space:]+balcón+[:space:]"
v4 = "[:space:]+tiene balcon+[:space:]"
v5 = "[:space:]+balcn+[:space:]"
v6 = "[:space:]+tiene balcn+[:space:]"
v7 = "[:space:]+bbq+[:space:]"
v8 = "[:space:]+tiene bbq+[:space:]"
v9 = "[:space:]+terraza+[:punct:]"
v10= "[:space:]+balcón+[:punct:]"
v11= "[:space:]+balcon+[:punct:]"
v12= "[:space:]+balcn+[:punct:]"
v13= "[:space:]+bbq+[:punct:]"

#nueva variable 
tt_sf <- tt_sf %>% 
  mutate(balcon = str_extract(string=tt_sf$description , pattern= paste0(v1,"|",v2,"|",v3,"|",v4,"|",v5,"|",v6,"|",
                                                                          v7,"|",v8,"|",v9,"|",v10,"|",v11,"|",v12,"|",v13)))
table(tt_sf$balcon)
table(is.na(tt_sf$balcon))

tt_sf <- tt_sf %>% 
  mutate(balcon = if_else( is.na(balcon)==TRUE,0,1)) 

summary(tt_sf$balcon) #32% de las propiedades tiene balcon o semejante


###duplex/penthouse/altillo##### extras 
u1 = "[:space:]+duplex+[:space:]"
u2 = "[:space:]+penthouse+[:space:]"
u3 = "[:space:]+pent house+[:space:]"
u4 = "[:space:]+pent-house+[:space:]"
u5 = "[:space:]+altillo+[:space:]"
u6 = "[:space:]+duplex+[:punct:]"
u7 = "[:space:]+penthouse+[:punct:]"
u8 = "[:space:]+pent house+[:punct:]"
u9 = "[:space:]+pent-house+[:punct:]"
u10 = "[:space:]+altillo+[:punct:]"

#nueva variable 
tt_sf <- tt_sf %>% 
  mutate(extras = str_extract(string=tt_sf$description , pattern= paste0(u1,"|",u2,"|",u3,"|",u4,"|",u5,"|",
                                                                          u6,"|",u7,"|",u8,"|",u9,"|",u10)))
table(tt_sf$extras)
table(is.na(tt_sf$extras))

tt_sf <- tt_sf %>% 
  mutate(extras = if_else( is.na(extras)==TRUE,0,1)) 

summary(tt_sf$extras) #3,4% de las propiedades son penthouse o semejante


###moderno/remodelado/renovado
t1 = "[:space:]+moderno+[:space:]"
t2 = "[:space:]+remodelado+[:space:]"
t3 = "[:space:]+renovado+[:space:]"
t4 = "[:space:]+moderno+[:punct:]"
t5 = "[:space:]+remodelado+[:punct:]"
t6 = "[:space:]+renovado+[:punct:]"

#nueva variable 
tt_sf <- tt_sf %>% 
  mutate(renov = str_extract(string=tt_sf$description , pattern= paste0(t1,"|",t2,"|",t3,"|",t4,"|",t5,"|",t6)))
table(tt_sf$renov)
table(is.na(tt_sf$renov))

tt_sf <- tt_sf %>% 
  mutate(renov = if_else( is.na(renov)==TRUE,0,1)) 

summary(tt_sf$renov) #7% de las propiedades son remodeladas o semejante


###vista/exterior
vi1 = "[:space:]+vista+[:space:]"
vi2 = "[:space:]+exterior+[:space:]"
vi3 = "[:space:]+vista+[:punct:]"
vi4 = "[:space:]+exterior+[:punct:]"

#nueva variable 
tt_sf <- tt_sf %>% 
  mutate(vista = str_extract(string=tt_sf$description , pattern= paste0(t1,"|",t2,"|",t3,"|",t4)))
table(tt_sf$vista)
table(is.na(tt_sf$vista))

tt_sf <- tt_sf %>% 
  mutate(vista = if_else( is.na(vista)==TRUE,0,1)) 

summary(tt_sf$vista) #4,9% de las propiedades tienen vista al exterior o semejante (y lo promocionan como algo destacable de la propiedad)



####tiene parqueadero
s1 = "[:space:]+parqueadero+[:space:]"
s2 = "[:space:]+parqueaderos+[:space:]"
s3 = "[:space:]+parqueadero+[:punct:]"
s4 = "[:space:]+parqueaderos+[:punct:]"

#nueva variable 
tt_sf <- tt_sf %>% 
  mutate(parq = str_extract(string=tt_sf$description , pattern= paste0(s1,"|",s2,"|",s3,"|",s4)))
table(tt_sf$parq)
table(is.na(tt_sf$parq))

tt_sf <- tt_sf %>% 
  mutate(parq = if_else( is.na(parq)==TRUE,0,1)) 

summary(tt_sf$parq) # 36% de las propiedades tienen parqueadero



#####tiene ascensor
r1 = "[:space:]+ascensor+[:space:]"
r2 = "[:space:]+ascensores+[:space:]"
r3 = "[:space:]+ascensor+[:punct:]"
r4 = "[:space:]+ascensores+[:punct:]"

#nueva variable 
tt_sf <- tt_sf %>% 
  mutate(ascen = str_extract(string=tt_sf$description , pattern= paste0(r1,"|",r2,"|",r3,"|",r4)))
table(tt_sf$ascen)
table(is.na(tt_sf$ascen))

tt_sf <- tt_sf %>% 
  mutate(ascen = if_else( is.na(ascen)==TRUE,0,1)) 

summary(tt_sf$ascen) #15% de las propiedades tienen ascensor


####info DANE#### >>> tendriamos que traer manzanas y no entiendo como se limpia el archivo de manzanas ##yo ya subi las manzanas, mira mas arriba

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










######################################################################################
######################################################################################
######################################################################################

##############     COPIA DE SEGURIDAD      ###########################################


###ESTO LO PONGO AL FINAL PARA GUARDAR UNA COPIA DEL WORKSPACE Y NO TENERLO QUE CORRER TODAS LAS VECES, PORQUE LO ESPACIAL SE DEMORA MUCHISIMO
#posdata: no lo guardo directamente en github porque pesa muchisimo y no me lo dejaria dar commit 

#Saving the workspace is essential when you work with scripts that take a long time to run 
#(for example simulation studies). This way, you can load the results without the need of running the script every time you open the script.

save.image(file = "C:/Users/Camila Cely/Documents/MECA/INTERSEMESTRALES/Big Data/problem set 3/backup_jul22_workspace.RData")

#As a consequence of saving your workspace, now you can load it so you won’t need to run the code to obtain those objects again.

load("C:/Users/Camila Cely/Documents/MECA/INTERSEMESTRALES/Big Data/problem set 3/backup_jul22_workspace.RData")



