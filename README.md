# opencv_lsd_line_cluster
该工程中是用与形状识别之直线检测(LSD)

linkedby:https://blog.csdn.net/qq_42189368/article/details/80432670
LSD官网（源码下载）：http://www.ipol.im/pub/art/2012/gjmr-lsd/

LSD官网（在线测试）：http://demo.ipol.im/demo/gjmr_line_segment_detector/


原作者作者检测四边形的思路主要为：

    直线检测
    直线聚类
    直线筛选
    交点计算
    交点排序
	
笔者经过实践的coding发现代码逻辑是:
	直线检测
    直线筛选(通过hsv的s饱和度进行直线的过滤)
	直线聚类(通过极坐标进行直线聚类)
    //后面两个步骤没有进行操作
	交点计算
    交点排序
	
代码中的知识点：
	rgb转hsv
	极坐标直线聚类
	通过极坐标求直线交点
	
工程中的测试图片为：idcard.jpg