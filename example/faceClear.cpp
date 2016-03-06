/*
The MIT License (MIT)

Copyright (c) 2015 Shiqi Yu
shiqi.yu@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <opencv.hpp>
#include <stdlib.h>
#include "facedetect-dll.h"
#include <time.h>
#include <string>
#include <io.h>
#include <iostream>
#pragma comment(lib,"libfacedetect.lib")

using namespace cv;

void clearFace(Mat& img, Rect box, Mat replace_img) {
	
	if (replace_img.rows < box.height || replace_img.cols < box.width) {
		resize(replace_img, replace_img, Size(box.height, box.width));
		replace_img.copyTo(img(box));
	}
	else {
		srand(time(NULL));
		int x = rand() % (replace_img.cols - box.width);
		int y = rand() % (replace_img.rows - box.height);
		replace_img(Rect(x, y, box.height, box.width)).copyTo(img(box));
	}
	
}
void getFiles(string filePath, vector<string>& fileNames) {
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(filePath).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(filePath).append("\\").append(fileinfo.name), fileNames);
			}
			else
			{
				fileNames.push_back(p.assign(filePath).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
int main(int argc, char* argv[])
{
	assert(argc == 4);
	//load an image and convert it to gray (single-channel)
	/*string input_directory = "F:\\C++Project\\GenNegSamples\\faceImgs";
	string replace_directory = "F:\\C++Project\\GenNegSamples\\replaceImgs";
	string output_directory = "F:\\C++Project\\GenNegSamples\\output";*/
	string input_directory(argv[1]);
	string replace_directory(argv[2]);
	string output_directory(argv[3]);
	vector<string> input_imgs;
	vector<string> replace_imgs;
	getFiles(input_directory, input_imgs);
	getFiles(replace_directory, replace_imgs);
	for (int i = 0; i < input_imgs.size(); i++) {
		//读取待处理图片的灰度图用于检测人脸
		Mat gray = imread(input_imgs[i], CV_LOAD_IMAGE_GRAYSCALE);
		//读取待处理图片的原图用于输出
		Mat raw = imread(input_imgs[i]);
		//随机选择一张替换图片
		srand(time(NULL));
		int pos_replace = rand() % replace_imgs.size();
		Mat replace_img = imread(replace_imgs[pos_replace]);
		if (gray.empty() || replace_img.empty())
		{
			continue;
		}
		//人脸检测
		int * pResults = NULL;
		pResults = facedetect_multiview_reinforce((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
			1.2f, 3, 24);
		//对待处理图像中的人脸全部进行替换
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 6 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			//int neighbors = p[4];
			Rect box(x, y, w, h);
			//rectangle(raw, box, Scalar(0, 255, 255, 0));
			clearFace(raw, box, replace_img);
			//printf("face_rect=[%d, %d, %d, %d], neighbors=%d\n", x, y, w, h, neighbors);
		}
		std::cout << input_imgs[i] << " done" << std::endl;
		//保存清理人脸后的图片
		int pos = input_imgs[i].find_last_of('\\');
		string save;
		if (pos == string::npos) {
			pos = input_imgs[i].find_last_of('/');
			string s(input_imgs[i].substr(pos + 1));
			save = output_directory + "/" + s;
		}
		else {
			string s(input_imgs[i].substr(pos + 1));
			save = output_directory + "\\" + s;
		}
		imwrite(save, raw);
			
	}
	
	/////////////////////////////////////////////
	//// multiview face detection 
	//// it can detection side view faces, but slower than the frontal face detection.
	////////////////////////////////////////////
	////!!! The input image must be a gray one (single-channel)
	////!!! DO NOT RELEASE pResults !!!
	//pResults = facedetect_multiview((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
	//	1.2f, 5, 24);
	//printf("%d faces detected.\n", (pResults ? *pResults : 0));

	////print the detection results
	//for (int i = 0; i < (pResults ? *pResults : 0); i++)
	//{
	//	short * p = ((short*)(pResults + 1)) + 6 * i;
	//	int x = p[0];
	//	int y = p[1];
	//	int w = p[2];
	//	int h = p[3];
	//	int neighbors = p[4];
	//	int angle = p[5];

	//	printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
	//}


	/////////////////////////////////////////////
	//// reinforced multiview face detection 
	//// it can detection side view faces, but slower than the frontal face detection.
	////////////////////////////////////////////
	////!!! The input image must be a gray one (single-channel)
	////!!! DO NOT RELEASE pResults !!!
	//pResults = facedetect_multiview_reinforce((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
	//	1.2f, 5, 24);
	//printf("%d faces detected.\n", (pResults ? *pResults : 0));

	////print the detection results
	//for (int i = 0; i < (pResults ? *pResults : 0); i++)
	//{
	//	short * p = ((short*)(pResults + 1)) + 6 * i;
	//	int x = p[0];
	//	int y = p[1];
	//	int w = p[2];
	//	int h = p[3];
	//	int neighbors = p[4];
	//	int angle = p[5];

	//	printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
	//}
	return 0;
}

