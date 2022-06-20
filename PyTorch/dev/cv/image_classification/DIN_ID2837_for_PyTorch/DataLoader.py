#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import torch
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class MyDataSet( torch.utils.data.Dataset):
    def __init__( self, data_path, user_map, material_map, category_map, max_length):

        user = []; material = []; category = []
        material_historical = []; category_historical = []
        material_historical_neg = []; category_historical_nge = []
        mask = []; sequential_length = []
        target = []

        with open( data_path, 'r') as fin:

            for line in fin:
                item = line.strip('\n').split('\t')
                if not item: continue

                user.append( user_map.get( item[1], 0 ) ) 
                material.append( material_map.get( item[2], 0 ) )
                category.append( category_map.get( item[3], 0 ) )

                material_historical_item = [0] * max_length
                temp = item[4].split("")
                if( len( temp) >= max_length): temp = temp[ -max_length:]
                for i, m in enumerate( temp):
                    material_historical_item[i] =  material_map.get( m, 0 ) 
                material_historical.append( material_historical_item)
                
                category_historical_item = [0] * max_length
                temp = item[5].split("")
                if( len( temp) >= max_length): temp = temp[ -max_length:]
                for i, c in enumerate( temp):
                    category_historical_item[i] =  category_map.get( c, 0 ) 
                category_historical.append( category_historical_item)

                temp = min( len(temp), max_length)
                mask_item = [1] * temp + [0] * ( max_length - temp)

                mask.append( mask_item)
                sequential_length.append( temp)

                target.append( int( item[0]))
        
        self.user = torch.tensor( user)

        self.material = torch.tensor( material)
        self.catetory = torch.tensor( category)

        self.material_historical = torch.tensor( material_historical)
        self.catetory_historical = torch.tensor( category_historical)

        self.mask = torch.tensor( mask).type( torch.FloatTensor)
        self.sequential_length = torch.tensor( sequential_length).type( torch.FloatTensor)

        self.target = torch.tensor( target).type( torch.FloatTensor)


    def __len__( self):
        return len( self.user)

    def __getitem__(self, index):
        if torch.is_tensor( index):
            index = index.tolist()

        user = self.user[ index]

        material_historical = self.material_historical[ index, :]
        category_historical = self.catetory_historical[ index, :]
        mask = self.mask[ index, :]
        sequential_length = self.sequential_length[ index]

        material = self.material[ index]
        category = self.catetory[ index]

        target = self.target[ index]

        return user, material_historical, category_historical, mask, sequential_length , \
            material, category, 0, 0, target