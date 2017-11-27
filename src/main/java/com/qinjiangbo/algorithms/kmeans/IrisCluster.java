package com.qinjiangbo.algorithms.kmeans;

import com.google.common.io.Resources;
import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;

/**
 * @date: 27/11/2017 6:58 PM
 * @author: qinjiangbo@github.io
 * @description:
 *      使用KMeans算法将Iris数据集进行聚类
 */
public class IrisCluster {

    public static void main(String[] args) {
        Instances instances;
        SimpleKMeans kMeans;

        try {
            URL url = Resources.getResource("data/iris.arff");
            // 读入样本数据
            File file = new File(url.getPath());
            ArffLoader loader = new ArffLoader();
            loader.setFile(file);
            instances = loader.getDataSet();

            // 初始化聚类器
            kMeans = new SimpleKMeans();
            // 设置聚类要得到的簇数
            kMeans.setNumClusters(3);
            // 开始进行聚类
            kMeans.buildClusterer(instances);

            // 打印聚类结果
            System.out.println(kMeans.preserveInstancesOrderTipText());
            System.out.println(kMeans.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
