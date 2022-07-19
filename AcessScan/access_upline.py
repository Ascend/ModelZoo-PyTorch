import re
import os
import sys
import json
import math
import glob
import argparse

class AccessCodeCheck(object):
	"""ModelZoo 门禁代码检查"""
	def __init__(self):
		# self.modelzoo_dir = "ModelZoo-PyTorch"
		self.modelzoo_dir = "modelzoo"
		self.args = self.init_args()
		self.prListFile = self.args.pr_filelist_dir
		self.fram_str = self.prListFile[:self.prListFile.index('pr_filelist.txt')]
		self.pr_filelist = self.get_result_dict()
		self.succResultList = []
		self.failResultList = []
		self.errorResultList = []
		# self.fileNameList = self.get_file_name_list()
		# self.fullPathList = self.get_full_path_list()

	def init_args(self):
		"""功能：读取通用参数"""
		parser = argparse.ArgumentParser()
		parser.add_argument('--pr_filelist_dir', type=str, default="./pr_filelist0.txt",
			help='model dirrectory of the pr_filelist')
		parser.add_argument('--linklisttxt', type=str, default="./link_list.txt",
			help='model dirrectory of the link_list')
		return parser.parse_args()

	def get_result_dict(self):
		"""
			功能：生成检查文件列表
		"""
		with open(self.prListFile, 'r') as f:
			content = f.read()
		content = content.split('\n')
		prfilelist = []
		for filename in content:
			fullname = os.path.join(self.fram_str, self.modelzoo_dir, filename)
			if os.path.exists(fullname):
				prfilelist.append(filename.strip('\n'))
		prfilelist = self.check_rawcode(prfilelist)
		return prfilelist

	def check_rawcode(self,filelist):
		"""
			功能：判断文件是否属于开源源码，开源源码不做检查
		"""
		no_raw_filelist = []
		raw_filelist = []
		for f in filelist:
			path_list = f.split("/")[:-1]
			for i, p in enumerate(path_list):
				file_path = os.path.join(self.fram_str, self.modelzoo_dir, *path_list[:i])
				path_list_files = os.listdir(file_path)
				if ".gitrawcode" in path_list_files:
					raw_filelist.append(f)
					break
		for f in filelist:
			if f not in raw_filelist:
				no_raw_filelist.append(f)
		return no_raw_filelist

	def file_size_check(self, prfile):
		"""
		功能：扫描文件大小，小于1M
		备注：暂时按照小于1M处理，后续细化，不同文件类型不同大小限制。
		"""
		prfilename = os.path.join(self.fram_str, self.modelzoo_dir, prfile)
		prfilesize = os.path.getsize(prfilename) / math.pow(1024, 2)
		if prfilesize < 1:
			self.succResultList.append("{}: filesize less than 1M, check succ!".format(prfile))
		else:
			self.failResultList.append("{}: filesize less than 1M, check fail!".format(prfile))

	def license_check(self, prfile):
		"""
			功能：判断.py/.cpp文件是否存在关键字LICENSE/license
		"""
		prfilename = os.path.join(self.fram_str, self.modelzoo_dir, prfile)
		if (prfilename[-3:] == ".py") or (prfilename[-4:] == ".cpp"):
			if prfilename[-11:] == "__init__.py":
				self.succResultList.append("{}: is __init__.py, no need to check license!".format(prfile))
			else:
				with open(prfilename, 'r') as f:
					content = f.read()
				content = content.lower()
				if ("license" in content) or ("licence" in content):
					self.succResultList.append("{}: contain LICENCE/licence, check succ!".format(prfile))
				else:
					self.failResultList.append("{}: not contain LICENCE/licence, check fail!".format(prfile))
		else:
			self.succResultList.append("{}: is not *.py/*.cpp, no need to check license!".format(prfile))

	def get_model_root(self, pr_file_str):
		"""
		功能：获取网络框架根目录
		"""
		modelRoot = ""
		if "/" not in pr_file_str:
			return modelRoot
		path_list = pr_file_str.split('/')
		path_list_len = len(path_list)
		model_parent_file = '.modelparant'
		while path_list_len > 1:
			tmp_path = os.path.join(modelRoot, *path_list[:path_list_len])
			path_list_len -= 1
			if os.path.isfile(os.path.join(self.fram_str, self.modelzoo_dir, tmp_path)):
				continue
			if model_parent_file in os.listdir(os.path.join(self.fram_str, self.modelzoo_dir, tmp_path)):
				return os.path.join(modelRoot, *path_list[:path_list_len])
		return ""

	def firstlevel_file_check(self, prfile):
		"""
			功能：判断网络框架根目录下是否包含必要的文件
				LICENCE，requirements.txt，modelzoo_level.txt， readme.md
		"""
		model_root = self.get_model_root(prfile)
		check_list = ["requirements.txt", "modelzoo_level.txt", "readme.md"]
		if model_root == "":
			self.succResultList.append("{}: at root path,no need to check, check succ!".format(prfile))
		if model_root != "":
			print("model_root: ",model_root)
			model_root_filelist_tmp = os.listdir(os.path.join(self.fram_str, self.modelzoo_dir, model_root))
			model_root_filelist = []
			for model_root_file in model_root_filelist_tmp:
				model_root_filelist.append(model_root_file.lower())
			for check_file in check_list:
				if check_file not in model_root_filelist:
					self.failResultList.append("{}: at model root path, does not contain {}, check fail!".format(model_root, check_file))
				else:
					self.succResultList.append("{}: at model root path, contain {}, check succ!".format(model_root, check_file))
			if ("license" in model_root_filelist) or ("licence" in model_root_filelist):
				self.succResultList.append("{}: at model root path, contain {}, check succ!".format(model_root, check_file))
			else:
				self.failResultList.append("{}: at model root path, does not contain {}, check fail!".format(model_root, check_file))

	def link_check(self, prfile):
		"""
			功能：检测文件内部是否包含内部链接
		"""
		prfilename = os.path.join(self.fram_str, self.modelzoo_dir, prfile)
		with open(self.args.linklisttxt, 'r') as f:
			content = f.read()
		linklist = content.split('\n')
		if ('readme' not in prfilename.lower()):
			with open(prfilename, 'r') as f:
				content = f.read()
			for link in linklist:
				if link in content:
					self.failResultList.append("{}: contain link[{}], check fail!".format(prfile, link))
				else:
					self.succResultList.append("{}: not contain link[{}], check succ!".format(prfile, link))
		else:
			self.succResultList.append("{}: is readme file, no need to check link, check succ!".format(prfile))


	def sensitive_content_check(self, prfile):
		"""
			功能：检测文件内部是否包含敏感信息(工号，http链接，黑名单)
		"""
		prfilename = os.path.join(self.fram_str, self.modelzoo_dir, prfile)
		with open(prfilename, 'r') as f:
			content = f.read()
		code_line_list = content.split('\n')
		with open(os.path.join(os.getcwd(),"upline_access_black_http.json"), 'r') as f:
			load_dict = json.load(f)
		blackhttp_list = load_dict["blackhttp"].split(',')
		for code_line in code_line_list:
			# 检查工号
			if ('0.00' not in code_line) or ('0.' not in code_line):
				if re.findall(r'[A-Za-z]00[1-9][\d]{4, 9}', code_line) or \
					re.findall(r'[A-Za-z]wx[1-9][\d]{4, 9}', code_line) or \
					re.findall(r'00[1-9][\d]{4, 9}', code_line):
					self.failResultList.append("{}: contain sensitive message in [{}], check fail!".format(prfile, code_line))
			# 检查http链接
			# if 'readme' not in prfile.lower():
			# 	if ('device_ip' in code_line) or ('server_id' in code_line):
			# 		continue
			# 	elif re.findall(r'http://\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
			# 				code_line) or \
			# 			re.findall(r'https://\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
			# 				code_line):
			# 		self.failResultList.append("{}: contain sensitive message in [{}], check fail!".format(prfile, code_line))
			# 检查黑名单
			else:
				for black_key in blackhttp_list:
					if 'contrib/' in str(prfile) and black_key == "pan.baidu.com":
						pass
					elif black_key in code_line:
						self.failResultList.append("{}: contain sensitive message in [{}], check fail!".format(prfile, code_line))

	def modelzoo_level_check(self, prfile):
		"""
			功能：检测modelzoo_level.txt文件
				1.是否存在
				2.文件内容是否合规
		"""
		level_dict = {}
		statuslist = ['FuncStatus', 'PerfStatus', 'PrecisionStatus']
		model_root = self.get_model_root(prfile)
		if model_root != '':
			model_dir = os.path.join(self.fram_str, self.modelzoo_dir, model_root)
			modelzoo_level_file = os.path.join(model_dir, 'modelzoo_level.txt')
			prlevelfile = os.path.join(model_root, 'modelzoo_level.txt')
			if os.path.exists(modelzoo_level_file):
				try:
					with open(modelzoo_level_file, 'r') as f:
						content = f.read()
					content = content.split('\n')
					for line in content:
						level_dict[line.split(':')[0]] = line.split(':')[1]
				except:
					print('The file modelzoo_level.txt in ' + model_root + 'can not open, please check it!')
				for status in statuslist:
					if status not in level_dict:
						self.failResultList.append("{}: The keyword of the[{}] is not exist in the level file, please check and add it, check fail!".format(prlevelfile, status))
					if level_dict[status] == '':
						self.failResultList.append("{}: The keyword of the[{}] is not null in the level file, please check and ehter the correct value, check fail!".format(prlevelfile, status))
			else:
				self.failResultList.append("{}: The level file is not exist, please check and add it, check fail!".format(prlevelfile))

	def file_word_ehck(self, prfile):
		"""
			功能：检查网络框架下test目录是否包含必要的train_full(1p or 8p)文件
				只检查pytorch下，acl下不检查
		"""
		model_root = self.get_model_root(prfile)
		if (model_root != '') and (model_root.startswith("PyTorch")):
			model_dir = os.path.join(self.fram_str, self.modelzoo_dir, model_root)
			test_dir = os.path.join(model_dir, 'test')
			if os.path.exists(test_dir):
				train_full = glob.glob(os.path.join(test_dir, "train*full*"))
				train_performance = glob.glob(os.path.join(test_dir, "train*performance*"))
				if not train_performance:
					self.failResultList.append("{}/test/: not contain the train_performance(1p or 8p) file, check fail!".format(model_root))
				if not train_full:
					self.failResultList.append("{}/test/: not contain the train_full(1p or 8p) file, check fail!".format(model_root))

	def check_result(self):
		"""
			功能：检查结果汇总输出
		"""
		print("++++++++++++++++++++ check pass log ++++++++++++++++++++")
		for l in self.succResultList:
			print(l)
		print("++++++++++++++++++++ check remove file log ++++++++++++++++++++")
		for l in self.errorResultList:
			print(l)
		print("++++++++++++++++++++ check no pass log ++++++++++++++++++++")
		for l in self.failResultList:
			print(l)
		
	def check_entrance(self):
		self.__init__()
		for pr_file in self.pr_filelist:
			filename = os.path.join(self.fram_str,self.modelzoo_dir,pr_file)
			if os.path.exists(filename) and os.path.isfile(filename):
				self.file_size_check(pr_file)
				self.license_check(pr_file)
				self.firstlevel_file_check(pr_file)
				self.link_check(pr_file)
				self.sensitive_content_check(pr_file)
				self.modelzoo_level_check(pr_file)
				self.file_word_ehck(pr_file)
			else:
				self.errorResultList.append("{}: The file does not exsist, please check if you remove this file!".format(pr_file))
		self.check_result()

def main():
	codeCheck = AccessCodeCheck()
	codeCheck.check_entrance()

if __name__ == '__main__':
	main()