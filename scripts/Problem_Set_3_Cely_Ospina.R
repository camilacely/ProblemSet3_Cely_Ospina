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


######################################################################################
######################################################################################
######################################################################################
                 
#MANZANAS BOGOTA

mbog <- read_sf ("stores/manzanasbogota/MGN_URB_MANZANA.shp")
st_crs(mbog)<-4326  

mchap <- st_crop (mbog, up_chapinero)

leaflet() %>% addTiles() %>% addPolygons(data=mchap , color="red") %>% addCircles(data=train_upchap)


#MANZANAS MEDELLIN





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
train_f = train_f %>% 
  mutate(area = ifelse(is.na(area)==T,
                       str_extract(string=train_f$description , pattern= 
                       paste0(x1,"|",x2,"|",x3,"|",x4,"|",x5,"|",x6,"|",x7,"|",x8,"|",x9,"|",x10,"|",
                              x11,"|",x12,"|",x13,"|",x14,"|",x15,"|",x16,"|",x17,"|",x18,"|",x19,"|",
                              x20,"|",x21,"|",x22,"|",x23,"|",x24,"|",x25,"|",x26,"|",x27)),
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
train_f = train_f %>% 
  mutate(bathrooms = ifelse(is.na(bathrooms)==T,
                       str_extract(string=train_f$description , pattern= 
                       paste0(y1,"|",y2,"|",y3,"|",y4,"|",y5,"|",y6,"|",y7,"|",y8,"|",y9,"|",y10,"|",y11,"|",y12,"|",y13)),
                       bathrooms))
table(is.na(train_f$bathrooms)) #20934 NA, imputamos 9140. las observaciones imputadas quedan con el ba, no se si eso afecte 

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
train_f = train_f %>% 
  mutate(nivel = str_extract(string=train_f$description , pattern= paste0(w1,"|",w2,"|",w3,"|",w4,"|",w5,"|",w6,"|",w7,"|",w8,"|",w9,"|",w10,"|",w11,"|",w12,"|",
                                                                          w13,"|",w14,"|",w15,"|",w16,"|",w17,"|",w18,"|",w19,"|",w20,"|",w21,"|",w22,"|",w23)))
table(train_f$nivel)
table(is.na(train_f$nivel)) #75611 NA, no se si valga la pena

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
train_f <- train_f %>% 
  mutate(nivel = str_extract(string=train_f$description , pattern= paste0(v1,"|",v2,"|",v3,"|",v4,"|",v5,"|",v6,"|",
                                                                          v7,"|",v8,"|",v9,"|",v10,"|",v11,"|",v12,"|",v13)))
table(train_f$nivel)
table(is.na(train_f$nivel))

train_f <- train_f %>% 
  mutate(nivel = if_else( is.na(area_total)==TRUE,0,1)) 

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
train_f <- train_f %>% 
  mutate(extras = str_extract(string=train_f$description , pattern= paste0(u1,"|",u2,"|",u3,"|",u4,"|",u5,"|",
                                                                          u6,"|",u7,"|",u8,"|",u9,"|",u10)))
table(train_f$extras)
table(is.na(train_f$extras))

train_f <- train_f %>% 
  mutate(extras = if_else( is.na(area_total)==TRUE,0,1)) 

###moderno/remodelado/renovado
t1 = "[:space:]+moderno+[:space:]"
t2 = "[:space:]+remodelado+[:space:]"
t3 = "[:space:]+renovado+[:space:]"
t4 = "[:space:]+moderno+[:punct:]"
t5 = "[:space:]+remodelado+[:punct:]"
t6 = "[:space:]+renovado+[:punct:]"

#nueva variable 
train_f <- train_f %>% 
  mutate(renov = str_extract(string=train_f$description , pattern= paste0(t1,"|",t2,"|",t3,"|",t4,"|",t5,"|",t6)))
table(train_f$renov)
table(is.na(train_f$renov))

train_f <- train_f %>% 
  mutate(renov = if_else( is.na(area_total)==TRUE,0,1)) 

###vista/exterior
t1 = "[:space:]+vista+[:space:]"
t2 = "[:space:]+exterior+[:space:]"
t3 = "[:space:]+vista+[:punct:]"
t4 = "[:space:]+exterior+[:punct:]"

#nueva variable 
train_f <- train_f %>% 
  mutate(vista = str_extract(string=train_f$description , pattern= paste0(t1,"|",t2,"|",t3,"|",t4)))
table(train_f$vista)
table(is.na(train_f$vista))

train_f <- train_f %>% 
  mutate(vista = if_else( is.na(area_total)==TRUE,0,1)) 

####tiene parqueadero
s1 = "[:space:]+parqueadero+[:space:]"
s2 = "[:space:]+parqueaderos+[:space:]"
s3 = "[:space:]+parqueadero+[:punct:]"
s4 = "[:space:]+parqueaderos+[:punct:]"

#nueva variable 
train_f <- train_f %>% 
  mutate(parq = str_extract(string=train_f$description , pattern= paste0(s1,"|",s2,"|",s3,"|",s4)))
table(train_f$parq)
table(is.na(train_f$parq))

train_f <- train_f %>% 
  mutate(parq = if_else( is.na(area_total)==TRUE,0,1)) 


#####tiene ascensor
r1 = "[:space:]+ascensor+[:space:]"
r2 = "[:space:]+ascensores+[:space:]"
r3 = "[:space:]+ascensor+[:punct:]"
r4 = "[:space:]+ascensores+[:punct:]"

#nueva variable 
train_f <- train_f %>% 
  mutate(ascen = str_extract(string=train_f$description , pattern= paste0(r1,"|",r2,"|",r3,"|",r4)))
table(train_f$ascen)
table(is.na(train_f$ascen))

train_f <- train_f %>% 
  mutate(ascen = if_else( is.na(area_total)==TRUE,0,1)) 



####info DANE#### >>> tendriamos que traer manzanas y no entiendo como se limpia el archivo de manzanas 

#por manzana calculo la mediana del numero de cuartos, nu de personas y el estrato
## load data
mnz_censo = import("http://eduard-martinez.github.io/data/fill-gis-vars/mnz_censo.rds")

## about data
browseURL("https://eduard-martinez.github.io/teaching/meca-4107/6-censo.txt")

## spatial join
house_censo = st_join(house_mnz,mnz_censo)
colnames(house_censo)






##filtrar chapinero y el poblado? > creo que al ser barrios con precios altos podriamos tener problemas al entrenar con todos los demas 
#Se podria filtrar como lo de imputar texto, buscar en title chapinero y el poblado

#Separar las bases entre Medellin y Bogota 
train_med <- train
train_med <- train_med [!(train_med$l3=="Bogotá D.C"),] #21.356 obs #medellin, (6.024 quitando NA) 

train_bog <- train
train_bog <- train_bog [(train_bog$l3=="Bogotá D.C"),] #86.211 obs #bogota, (27.035 quitando NA)



table(train_f$title)

ch = "chapinero"## pattern
p = "el poblado"
p2= "poblado"




chapinero <- getbb(place_name = "UPZ Chapinero, Bogota", 
                   featuretype = "boundary:administrative", 
                   format_out = "sf_polygon") %>% .$multipolygon

leaflet() %>% addTiles() %>% addPolygons(data=chapinero)

house_censo = st_join(house_mnz,mnz_censo)
colnames(house_censo)

view(train_f$description)



###PREDICTORS COMING FROM EXTERNAL SOURCES

