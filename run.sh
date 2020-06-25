#!/bin/bash

num_linhas=$(cat parametros | wc -l)
((num_linhas--))
echo Numero de linhas/Simulacao total: $num_linhas

cont=1
while [ $cont -le $num_linhas ]
do
	echo "Botando para rodar o $cont"
	python3 main.py $cont > log.$cont &
	((cont++))
	sleep 1s
done


