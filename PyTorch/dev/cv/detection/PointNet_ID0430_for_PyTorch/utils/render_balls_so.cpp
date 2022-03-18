/*
* BSD 3-Clause License
*
* Copyright (c) 2017 xxxx
* All rights reserved.
* Copyright 2021 Huawei Technologies Co., Ltd
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ============================================================================
*/

#include <cstdio>
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;

struct PointInfo{
	int x,y,z;
	float r,g,b;
};

extern "C"{

void render_ball(int h,int w,unsigned char * show,int n,int * xyzs,float * c0,float * c1,float * c2,int r){
	r=max(r,1);
	vector<int> depth(h*w,-2100000000);
	vector<PointInfo> pattern;
	for (int dx=-r;dx<=r;dx++)
		for (int dy=-r;dy<=r;dy++)
			if (dx*dx+dy*dy<r*r){
				double dz=sqrt(double(r*r-dx*dx-dy*dy));
				PointInfo pinfo;
				pinfo.x=dx;
				pinfo.y=dy;
				pinfo.z=dz;
				pinfo.r=dz/r;
				pinfo.g=dz/r;
				pinfo.b=dz/r;
				pattern.push_back(pinfo);
			}
	double zmin=0,zmax=0;
	for (int i=0;i<n;i++){
		if (i==0){
			zmin=xyzs[i*3+2]-r;
			zmax=xyzs[i*3+2]+r;
		}else{
			zmin=min(zmin,double(xyzs[i*3+2]-r));
			zmax=max(zmax,double(xyzs[i*3+2]+r));
		}
	}
	for (int i=0;i<n;i++){
		int x=xyzs[i*3+0],y=xyzs[i*3+1],z=xyzs[i*3+2];
		for (int j=0;j<int(pattern.size());j++){
			int x2=x+pattern[j].x;
			int y2=y+pattern[j].y;
			int z2=z+pattern[j].z;
			if (!(x2<0 || x2>=h || y2<0 || y2>=w) && depth[x2*w+y2]<z2){
				depth[x2*w+y2]=z2;
				double intensity=min(1.0,(z2-zmin)/(zmax-zmin)*0.7+0.3);
				show[(x2*w+y2)*3+0]=pattern[j].b*c2[i]*intensity;
				show[(x2*w+y2)*3+1]=pattern[j].g*c0[i]*intensity;
				show[(x2*w+y2)*3+2]=pattern[j].r*c1[i]*intensity;
			}
		}
	}
}

}//extern "C"
