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


###################################################################
#QUE PASARIA SI ELIMINAMOS TODOS LOS MISSINGS DE TRAIN (LA ORIGINAL)

train_min <- train %>% drop_na(c("surface_total")) #queda de 27.722 observaciones #de banos quedamos con 165 missings, creo que podemos eliminarlos tambien
summary(train_min) 

train_min <- train_min %>% drop_na(c("bathrooms")) #queda de 27.557 obs y sin NAs en bathrooms
summary(train_min)

#ahora, como se comporta surface_total en esta submuestra

summary(train_min$surface_total) #el valor minimo es de 11 y el maximo es de 108.800, querriamos eliminarnos a ambos

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#11      70     108     173     189  108800 

#En todo caso, lo importante es ver que realmente estariamos perdiendo muchas observaciones, por lo cual por ahora no usaremos train_min
####################################################################




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






boxplot(train_min$surface_total, 
        ylab = "Area total"
)

#entonces vamos a reducirle colas

lower_area<-0.01
upper_area<-0.99

lower_bound_atr<- quantile(train_min$surface_total, lower_area) 
lower_bound_atr #35 m2

upper_bound_atr <- quantile(train_min$surface_total, upper_area) 
upper_bound_atr #936 m2


train_min <- train_min %>% subset(surface_total >= lower_bound_atr) 
train_min <- train_min %>% subset(surface_total <= upper_bound_atr) #queda de 27.035 obs


boxplot(train_min$surface_total, 
        ylab = "Area total"
)


#veamos como queda el summary con esta nueva distribucion

train_min %>%
  select(price, start_date, l2, bedrooms, bathrooms, surface_total, property_type) %>%
  tbl_summary()

#de aqui (train_min) obtenemos a grandes rasgos lo siguiente=

# PRECIO PROMEDIO = 445 millones #al quitar las casas mas grandes, redujimos un poco el precio promedio
# DEPARTAMENTOS = 77% estan en Cundinamarca y 23% en Antioquia ###
# CUARTOS = En promedio hay 3
# BANOS = En promedio hay 2 (ya no hay missings)
# AREA TOTAL = Promedio 107 m2 (ya no hay missings)
# TIPO VIVIENDA = 67% apartamento, 33% casa

hist(train_min$price)


#voy a crear un subset por ciudad

train_med <- train_min
train_med <- train_med [!(train_med$l3=="Bogotá D.C"),] #6.024 obs #medellin

train_bog <- train_min
train_bog <- train_bog [(train_bog$l3=="Bogotá D.C"),] #27.035 obs #bogota

#espacializar

bogota <- st_as_sf(x=train_bog,coords=c("lon","lat"),crs=4326)
class(bogota)
bogota

leaflet() %>% addCircleMarkers(data=bogota) #visualizacion dinamica

ggplot()+
  geom_sf(data=bogota) +
  theme_bw() +
  theme(axis.title =element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size=6))    #visualizacion a modo de plot 

#vemos en las dos salidas anteriores que hay datos para toda la ciudad, no solamente para Chapinero


#voy a traer la informacion de las upz

upla<-read_sf("stores/upla/UPla.shp") #totalidad de la ciudad

up_chapinero <- upla %>% filter(UPlNombre
                                %in%c("EL REFUGIO", "PARDO RUBIO", "CHICO LAGO", "CHAPINERO")) #aqui agarro las upz de chapinero (localidad)

bar <- opq(bbox = st_bbox(up_chapinero)) %>% #cuando hacemos esta st_bbox de solo chapinero, ya no nos salen los puntos de toda la ciudad sino solo dentro de esa area
  add_osm_feature(key = "amenity", value = "bar") %>%
  osmdata_sf() %>% .$osm_points %>% select(osm_id,name)

bar %>% head() 

leaflet() %>% addTiles() %>% addCircleMarkers(data=bar , col="red")  #notar que aqui esta hecho con bares a modo de ejemplo




######################################################################################
######################################################################################
######################################################################################
                 
###PREDICTORS COMING FROM EXTERNAL SOURCES



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

