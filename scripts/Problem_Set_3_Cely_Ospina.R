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
       leaflet) #por ahora llame todas las del problem set 2

predict<- stats::predict  #con esto soluciono el problema de que haya mas de una libreria con este comando


#####################
##cargar los datos #
#####################

##Establecer el directorio

#setwd
#setwd("C:/Users/SARA/Documents/ESPECIALIZACIÓN/BIG DATA/GITHUB/ProblemSet2_Cely_Ospina")
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
#ademas no estoy segura de que quieren decir las fechas
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
#view(train$area_descubierta)  #la verdad se generan 96.625 NAs y valores negativos #las voy a dejar para que no corran porque por ahora creo que esta variable no es necesaria

#creo que lo mejor sera guiarnos por surface_total

var_lab(train$surface_total) = "Area total"


#dinero
summary(train$currency) #aqui nos dice que estan medidas en COP
view(train$currency)
sum(train$currency == "USD") #sale 0  #vemos que no hay valores en dolares
sum(train$currency == "COP") #107567  #la totalidad de los precios estan en COP


#anuncio
view(train$title) #aqui sale el titulo de la oferta, ejemplo= "hermosa casa en venta"
var_lab(train$title) = "Titulo anuncio"

view(train$description) #aqui dice los detalles, ejemplo = barrio, "muy iluminado", "moderno", etc7
var_lab(train$description) = "Descripcion anuncio"

#tipo de propiedad
view(train$property_type) #esta nos dice si es casa o apartamento
var_lab(train$property_type) = "Tipo de propiedad"

#venta o arriendo
view(train$operation_type) #aqui nos dice si es venta o arriendo
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


#QUE PASA SI ELIMINAMOS TODOS LOS MISSINGS

train_min <- train %>% drop_na(c("surface_total")) #queda de 27.722 observaciones #de banos quedamos con 165 missings, creo que podemos eliminarlos tambien
summary(train_min) 

train_min <- train_min %>% drop_na(c("bathrooms")) #queda de 27.557 obs y sin NAs en bathrooms
summary(train_min)

#ahora, como se comporta surface_total en esta submuestra

summary(train_min$surface_total) #el valor minimo es de 11 y el maximo es de 108.800, querriamos eliminarnos a ambos

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#11      70     108     173     189  108800 


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

#veamos algunas relaciones basicas entre variables

ggplot(data = train_min , mapping = aes(x = surface_total , y = price))+
  geom_point(col = "tomato" , size = 0.75) #aqui se evidencia que a mayor area, mayor precio


ggplot(data = train_min , mapping = aes(x = bathrooms , y = price))+
  geom_point(col = "tomato" , size = 0.75) #al principio entre mas banos mas precio, pero luego esto se empieza a comportar distinto cuando hay ya demasiados banos, como mas de 5
#resta la duda de si deberiamos eliminar estos outliers de bano
summary(train_min$bathrooms)

ggplot(data = train_min , mapping = aes(x = bedrooms , y = price))+
  geom_point(col = "tomato" , size = 0.75) #pasa lo mismo que con bathrooms, aumenta el precio como hasta 5 bedrooms y luego va disminuyendo
summary(train_min$bedrooms)

##Nota= esto ya lo estoy sacando con train_min , que es la base sin missings, si en algun momento quisieramos imputarle valores a estos missings habria que modificar esta parte

###PREDICTORS COMING FROM EXTERNAL SOURCES

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

ggplot()+
  geom_sf(data=upla
          %>% filter(UPlNombre
                     %in%c("EL REFUGIO","SAN ISIDRO - PATIOS", "PARDO RUBIO", "CHICO LAGO", "CHAPINERO")), fill = NA) +
  geom_sf(data=bogota, col="red") +
  
  theme_bw() +
  theme(axis.title =element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size=6))   #aqui nos salen todos los puntos de la ciudad, no los de chapinero




#lo que tenemos que hacer es determinar si quedan dentro de los poligonos que nos importan o no

up_chapinero <- upla %>% filter(UPlNombre
                                %in%c("EL REFUGIO", "PARDO RUBIO", "CHICO LAGO", "CHAPINERO"))

plot (bogota$geometry)
plot (upla)
plot(up_chapinero)

ggplot()+
  geom_sf(data=up_chapinero,
          fill = NA) +
  geom_sf(data=bogota, col="red") 
  theme_bw() +
  theme(axis.title =element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size=6))



bogota[up_chapinero, , op = st_intersects]

prueba1 = bogota |>
  st_filter(y = up_chapinero, .predicate = st_intersects)

st_intersection (bogota , up_chapinero)

good_points <- st_filter(bogota$geometry, up_chapinero$geometry)  ##NOTA = NO HE PODIDO HACER LA INTERSECCION DE PUNTOS EN EL POLIGONO DE CHAPINERO! PENDIENTE REVISAR
                 
                 

###PREDICTORS COMING FROM DESCRIPTION






