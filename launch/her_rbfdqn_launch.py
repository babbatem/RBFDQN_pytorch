import itertools
import argparse
import datetime
import os
import sys
import re
import time
import numpy as np
import argparse

def filldict(listKeys, listValues):
	mydict = {}
	for key, value in zip(listKeys, listValues):
		 mydict[key] = value
	return mydict

def generate_script_body(param_dict):
	script_body=\
'''#!/bin/bash

#SBATCH --time=8:00:00

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -J {}
#SBATCH --mem=12G
#SBATCH -p gpu --gres=gpu:1

#SBATCH -o {}-%j.out
#SBATCH -e {}-%j.out

cd /users/babbatem/
source .bashrc
source load_mods.sh

cd RBFDQN_pytorch
python3 HER_RBFDQN.py {} {}

'''
	script_body=script_body.format(param_dict['name'],
								   param_dict['name'],
								   param_dict['name'],
								   param_dict['env'],
								   param_dict['seed'])
	return script_body

def submit(param_dict):
	script_body = generate_script_body(param_dict)

	objectname = param_dict['env'] + '-' \
			   + str(param_dict['seed'])

	jobfile = "scripts_her/{}/{}".format(param_dict['name'], objectname)
	with open(jobfile, 'w') as f:
		f.write(script_body)
	cmd="sbatch {}".format(jobfile)
	os.system(cmd)
	return 0

def main(args):

	KEYS = ['seed', 'env', 'name']
	SEEDS = np.arange(3)

	os.makedirs('scripts_her/%s' % args.exp_name, exist_ok=True)

	k=0
	for j in range(len(SEEDS)):
		element = [SEEDS[j],
				   args.env,
				   args.exp_name]

		param_dict = filldict(KEYS, element)
		submit(param_dict)
		k+=1
	print(k)

if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-t', '--test', action='store_true', help='don\'t submit, just count')
	parser.add_argument('-n', '--exp-name', required=True, type=str, help='parent directory for jobs')
	parser.add_argument('-e', '--env', type=str, help='numeric, e.g. 60')
	args=parser.parse_args()
	main(args)
