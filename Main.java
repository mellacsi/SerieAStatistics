import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.lang.reflect.Array;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.*;

public class Main {
    public static ArrayList<CSVRecord> completeList = null;
    public static void main(String[] args){

        try {
            //ANALISI ESPLORATIVA
            //completeList = readFile();
            //generateFaulsVsYellows();
            //generateShotVsGoals();
            //generateGoalsAtHomeVsGoalsAway();
            //generateGoalsVsShot();
            //generateFaulsVsGoals();


            //PROBLEMI DI CLASSIFICAZIONE
            //genera file weka per classificare
            //generateWeka();
            //classifica e disegna
            //fourClassification();
            //randomClassification();
            kNN();
            //Pseudocode kNN Classificator (non ancora funzionante al 100)
            //mykNN();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }


//-------------------------------------------------------------------------------------------------------------------------------------------------------------//
    ////////////////////////////////////////////////////////////////CLASSIFICATORI/////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------------------------------------------------------------------------------------//

    //mio pseudocode per knn
    private static void mykNN() throws Exception {
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(
                "./src/resources/SerieA2016.arff");
        Instances data = ds.getDataSet();
        data.setClass(data.attribute("resultFT"));
        // Ora data contiene le mie istanze

        int[] keep = {data.attribute("resultFT").index(), data.attribute("pctHShot").index(), data.attribute("differenceGoalHT").index()};
        Remove remove = new Remove();
        remove.setInvertSelection(true);
        remove.setAttributeIndicesArray(keep);
        remove.setInputFormat(data);
        data = weka.filters.Filter.useFilter(data, remove);

        // Mescolo l'ordine delle istanze, altrimenti c'e' il rischio che non siano ben distribuite nel successivo split
        data.randomize(new Random());

        // Definisco e implemento lo split in totale sono 380
        int nTrain = 300;
        int nTest = data.numInstances()-nTrain;
        Instances dataTrain = new Instances(data, 0, nTrain);
        Instances dataTest = new Instances(data, nTrain, nTest);

        ArrayList<Double> xList = new ArrayList<>();
        ArrayList<Double> yList = new ArrayList<>();
        for (Instance instance : data) {
            xList.add(instance.value(1));
            yList.add(instance.value(2));
        }

        makeDoubleScatterPlot(xList, yList, "%Shots", "Difference Goals HT");

        ArrayList<Double> nkkX = new ArrayList<>();
        ArrayList<Double> percCorrectsY = new ArrayList<>();

        int times = 0;
        for (int z = 0; z < 380; z++) {
            if (z % 2 != 0) {
                times = z;
                Map<Instance, Double> predicted = new HashMap<>();

                for (Instance pivot : dataTest) {
                    //distanze dal più vicino al più lontano
                    TreeMap<Double, Instance> distances = new TreeMap();
                    for (Instance instance : dataTrain) {
                        if (pivot != instance) {
                            distances.put(Math.sqrt(Math.pow(pivot.value(1) - instance.value(1), 2)
                                    + Math.pow(pivot.value(2) - instance.value(2), 2)), instance);
                        }
                    }
                    int class0 = 0;
                    int class1 = 0;
                    int i = 1;
                    Set keys = distances.keySet();
                    //mantengo l'ordine dal minore al maggiore
                    for (Iterator j = keys.iterator(); j.hasNext(); ) {
                        Double key = (Double) j.next();
                        Instance value = (Instance) distances.get(key);
                        //System.out.println("K: " + key + " V: " + value);

                        if (value.value(0) == 0.)
                            class0++;
                        else
                            class1++;

                        if (i == times)
                            break;
                        i++;
                    }

                    if (class0 > class1) {
                        //System.out.println("Prev: H");
                        predicted.put(pivot, 0.);
                    } else {
                        //System.out.println("Prev: A");
                        predicted.put(pivot, 1.);
                    }
                }

                double correct = 0;
                double value;
                int i = 0;

                for (Map.Entry<Instance, Double> entry : predicted.entrySet()) {
                    i = 0;
                    Instance real = dataTest.instance(i);
                    Instance predict = entry.getKey();
                    while (real != predict) {
                        i++;
                        real = data.instance(i);
                    }

                    //System.out.println("Estimated: " + entry.getValue() + " Real: " + data.instance(i).value(0));
                    if (entry.getValue() == data.instance(i).value(0))
                        correct++;
                }
                System.out.println("KNN classifier");
                System.out.println("% Correct: " + correct / predicted.size());
                nkkX.add(times + 0.);
                percCorrectsY.add(correct / predicted.size());
            }
        }
        makeDoubleScatterPlot(nkkX, percCorrectsY, "N", "%Corrects");
    }

    //usa un knn e prova diversi valori di n
    private static void kNN() throws Exception {
        // ### CARICO I DATI ####
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(
                "./src/resources/SerieA2016.arff");
        Instances data = ds.getDataSet();
        data.setClass(data.attribute("resultFT"));
        // Ora data contiene le mie istanze

        int[] keep = { data.attribute("resultFT").index(), data.attribute("pctHShot").index() ,data.attribute("differenceGoalHT").index()};
        Remove remove = new Remove();
        remove.setInvertSelection(true);
        remove.setAttributeIndicesArray(keep);
        remove.setInputFormat(data);
        data = weka.filters.Filter.useFilter(data, remove);

        // Mescolo l'ordine delle istanze, altrimenti c'e' il rischio che non siano ben distribuite nel successivo split
        data.randomize(new Random());

        // Definisco e implemento lo split in totale sono 380
        int nTrain = 300;
        int nTest = data.numInstances()-nTrain;
        Instances dataTrain = new Instances(data, 0, nTrain);
        Instances dataTest = new Instances(data, nTrain, nTest);

        int nValue = 0;
        ArrayList<Double> nkkX = new ArrayList<>();
        ArrayList<Double> percCorrectsY = new ArrayList<>();

        for (int p = 0; p < 160; p++) {
            if (p % 2 != 0) {
                nValue = p;
                Classifier cls = new IBk(nValue);

                // Addestro il classificatore
                cls.buildClassifier(dataTrain);
                Evaluation eval = new Evaluation(dataTrain); // la istanzio con i dati di training
                double[] evaluation = eval.evaluateModel(cls, dataTest); // richiedo di valutare il classificatore sui dati di test
                System.out.println(eval.toSummaryString());

                System.out.println("KNN classifier");
                System.out.println("% Correct: " + eval.correct() / evaluation.length);
                nkkX.add(p + 0.);
                percCorrectsY.add(eval.correct() / evaluation.length);
            }
        }
        makeDoubleScatterPlot(nkkX, percCorrectsY, "Threshold", "Accuracy");
    }

    private static void randomClassification() throws Exception {
        // ### CARICO I DATI ####
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(
                "./src/resources/SerieA2016.arff");

        ArrayList<double[]> allPredictions = new ArrayList<>();
        Instances dataTest = null;
        Classifier cls = new RandomForest();
        for (int j = 0; j < 3; j++) {
            Instances data = ds.getDataSet();
            data.setClass(data.attribute("resultFT"));
            int[] keep = null;
            if(j == 0){
                //pct tiri fatti da squadra in casa FT e differenza reti HT
                //le nuove classi sono H (che comprende pareggi)e A
                int[] keeps = {  data.attribute("resultFT").index(), data.attribute("differenceGoalHT").index(), data.attribute("pctHShot").index() };
                keep = keeps;
            } else if(j == 1){
                int[] keeps = {  data.attribute("resultFT").index(), data.attribute("differenceGoalHT").index()};
                keep = keeps;
            } else if(j == 2){
                int[] keeps = {  data.attribute("resultFT").index(), data.attribute("pctHShot").index() };
                keep = keeps;
            }

            Remove remove = new Remove();
            remove.setInvertSelection(true);
            remove.setAttributeIndicesArray(keep);
            remove.setInputFormat(data);
            data = weka.filters.Filter.useFilter(data, remove);

            //non mischio per tenere come riferimento sempre lo stesso test set tanto i risultati delle partite di un campionato
            //sono già mischiati e quindi non rispettano un certo ordine per vinte perse pareggiate
            // Definisco e implemento lo split in totale sono 380
            int nTrain = 200;
            int nTest = data.numInstances()-nTrain;
            Instances dataTrain = new Instances(data, 0, nTrain);
            dataTest = new Instances(data, nTrain, nTest);

            // Addestro il classificatore
            cls.buildClassifier(dataTrain);

            // Guardo dentro al classificatore
            //System.out.println(((RandomForest) cls).toString());

            // In alternativa posso usare l'apposita classe
            Evaluation eval = new Evaluation(dataTrain); // la istanzio con i dati di training
            //classi stimate dei test
            double[] evaluation = eval.evaluateModel(cls, dataTest); // richiedo di valutare il classificatore sui dati di test
            //for (double val : evaluation) {
            //System.out.print(val + "; ");
            //}
            System.out.println();
            System.out.println("-----------------------------------------------------");
            System.out.println(eval.toSummaryString());
            System.out.println("-----------------------------------------------------");
            double[] predictions = new double[dataTest.size()];
            for (int i = 0; i < dataTest.numInstances(); i++) {
                Instance testInstance = dataTest.instance(i);
                double[] predDist = cls.distributionForInstance(testInstance);
                predictions[i] = predDist[1];
            }
            allPredictions.add(predictions);

        }

        calculateROC(dataTest.attributeToDoubleArray(dataTest.attribute("resultFT").index()), allPredictions);
    }


    //usa diversi classificatori per i miei dati
    private static void fourClassification() throws Exception {
        // ### CARICO I DATI ####
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(
                "./src/resources/SerieA2016.arff");
        Instances data = ds.getDataSet();
        data.setClass(data.attribute("resultFT"));
        // Ora data contiene le mie istanze

        //pct tiri fatti da squadra in casa FT e differenza reti HT
        //le nuove classi sono H (che comprende pareggi)e A
        int[] keep = {  data.attribute("resultFT").index(), data.attribute("differenceGoalHT").index(), data.attribute("pctHShot").index() };
        Remove remove = new Remove();
        remove.setInvertSelection(true);
        remove.setAttributeIndicesArray(keep);
        remove.setInputFormat(data);
        data = weka.filters.Filter.useFilter(data, remove);

        // Mescolo l'ordine delle istanze, altrimenti c'e' il rischio che non siano ben distribuite nel successivo split
        data.randomize(new Random());

        // Definisco e implemento lo split in totale sono 380
        int nTrain = 200;
        int nTest = data.numInstances()-nTrain;
        Instances dataTrain = new Instances(data, 0, nTrain);
        Instances dataTest = new Instances(data, nTrain, nTest);

        /*
        //Prova con classificatore che risponde a caso:
        double[] dummyPrediction = new double[dataTest.size()];
        for (int i=0;i<dataTest.numInstances();i++) {
            Instance testInstance = dataTrain.instance(i);
            if(Math.random() >= 0.5)
                dummyPrediction[i] = 1.;
            else
                dummyPrediction[i] = 0.;
        }
        double correct = 0;
        double value;
        for (int i=0;i<dummyPrediction.length;i++) {
            if(dataTest.instance(i).value(0) == 'H')
                value = 1.;
            else
                value = 0.;
            if(dummyPrediction[i] == value)
                correct++;
        }
        System.out.println("Random classifier");
        System.out.println("% Correct: " + correct/dummyPrediction.length);
        */
        ArrayList<double[]> allPredictions = new ArrayList<>();
        Classifier cls = null;
        for (int j = 0; j < 4; j++) {
            if(j == 0)
                cls = new NaiveBayes();
            else if(j == 1)
                cls = new PART();
            else if(j == 2)
                cls = new IBk(91);
            else if(j == 3)
                cls = new RandomForest();

            // Addestro il classificatore
            cls.buildClassifier(dataTrain);

            // Guardo dentro al classificatore
            //System.out.println(((RandomForest) cls).toString());

            // In alternativa posso usare l'apposita classe
            Evaluation eval = new Evaluation(dataTrain); // la istanzio con i dati di training
            //classi stimate dei test
            double[] evaluation = eval.evaluateModel(cls, dataTest); // richiedo di valutare il classificatore sui dati di test
            //for (double val : evaluation) {
            //System.out.print(val + "; ");
            //}
            System.out.println();
            System.out.println("-----------------------------------------------------");
            System.out.println(eval.toSummaryString());
            System.out.println("-----------------------------------------------------");
            double[] predictions = new double[dataTest.size()];
            for (int i = 0; i < dataTest.numInstances(); i++) {
                Instance testInstance = dataTest.instance(i);
                double[] predDist = cls.distributionForInstance(testInstance);
                predictions[i] = predDist[1];
            }
            allPredictions.add(predictions);
        }

        calculateROC(dataTest.attributeToDoubleArray(dataTest.attribute("resultFT").index()), allPredictions);
    }

    //dati i valori stimati e quelli originali calcola la ROC cambiando threashold fra 0 e 100 poi la disegna
    private static void calculateROC(double[] y, ArrayList<double[]> yHat){
        //y = valori veri della classe
        //yHat = % che l'istanza sia di classe true, dal classificatore
        ArrayList<ArrayList<Double>> alltpr = new ArrayList<>();
        ArrayList<ArrayList<Double>> allfpr = new ArrayList<>();

        for (int i1 = 0; i1 < yHat.size(); i1++) {
            double[] classification = yHat.get(i1);
            ArrayList<Double> tpr = new ArrayList<>();
            ArrayList<Double> fpr = new ArrayList<>();
/*
        for (double val: y ) {
            System.out.print(" " + val);
        }
        //System.out.println("---------------------------------");
        for (double val: yHat ) {
            System.out.print(" " + val);
        }
        System.out.println();
*/
            double threshold = 0.;
            //System.out.println("y size: " + y.length + " yHat size: " + yHat.length);
            ArrayList<Double> threshX = new ArrayList<>();
            ArrayList<Double> accY = new ArrayList<>();


            for (int j = 0; j < 101; j++) {
                double[] yPredict = new double[yHat.get(i1).length];
                double[][] confusionMat = new double[2][2];
                for (int i = 0; i < yHat.get(i1).length; i++) {
                    //calcolo nuova stima
                    if (yHat.get(i1)[i] >= threshold) {
                        yPredict[i] = 1.;
                    } else {
                        yPredict[i] = -1.;
                    }

                    //calcolo somme tp fp tn fn
                    if (yPredict[i] == 1. && y[i] == 1.) {
                        confusionMat[1][1]++;
                    } else if (yPredict[i] == 1. && y[i] == 0) {
                        confusionMat[0][1]++;
                    } else if (yPredict[i] == -1. && y[i] == 1.) {
                        confusionMat[1][0]++;
                    } else {
                        confusionMat[0][0]++;
                    }
                }
                threshX.add(threshold);
                //(confusionMat[0][0] + confusionMat[1][1])/(confusionMat[0][0] + confusionMat[0][1] + confusionMat[1][0] + confusionMat[1][1])
                accY.add((confusionMat[0][0] + confusionMat[1][1]) / yHat.get(i1).length);
                threshold += 0.01;
            /*
            System.out.print("Classe pred -> \t");
            for(int z=0;z<confusionMat.length;z++)
                System.out.print(""+z+"\t");
            System.out.println();
            for(int k=0;k<confusionMat.length;k++) {
                System.out.print("Classe vera "+k+": \t");
                for(int l=0;l<confusionMat.length;l++)
                    System.out.print(""+confusionMat[k][l]+"\t");
                System.out.println();
            }
            System.out.println("----------------------------------------------");
            */
                //calcolo tpr e fpr
                tpr.add(confusionMat[1][1] / (confusionMat[1][0] + confusionMat[1][1]));
                fpr.add(confusionMat[0][1] / (confusionMat[0][0] + confusionMat[0][1]));
            }


            System.out.println("FPR: " + fpr.size());
            //System.out.println(fpr);
            System.out.println("TPR: " + tpr.size());
            //System.out.println(tpr);
            allfpr.add(fpr);
            alltpr.add(tpr);
        }
            //disegna ROC con i tpr e fpr calcolati
            makeROC(allfpr, alltpr, "%FP", "%TP");
            //makeDoubleScatterPlot(threshX, accY, "Threshold", "Accuracy");
    }

    //genera file .arff per weka
    public static void generateWeka() throws IOException {
        ArrayList<String> teams = new ArrayList<>(Arrays.asList("Atalanta", "Bologna", "Cagliari", "Chievo", "Crotone", "Empoli", "Fiorentina", "Genoa", "Inter", "Juventus", "Lazio", "Milan", "Napoli", "Palermo", "Pescara", "Roma", "Sampdoria", "Sassuolo", "Torino", "Udinese"));
        ArrayList<String> results = new ArrayList<>(Arrays.asList("H", "A"));
        ArrayList<Attribute> atts = new ArrayList<>();
        Attribute attHomeTeam = new Attribute("homeTeam", teams);
        Attribute attAwayTeam = new Attribute("awayTeam", teams);
        Attribute attFTHG = new Attribute("homeGoalFT"); // Attributo numerico
        Attribute attFTAG = new Attribute("awayGoalFT"); // Attributo numerico
        Attribute attFTR = new Attribute("resultFT", results); // Attributo H D A
        Attribute attHTHG = new Attribute("homeGoalHT"); // Attributo numerico
        Attribute attHTAG = new Attribute("awayGoalHT"); // Attributo numerico
        //Attribute attHTR = new Attribute("resultHT");
        Attribute attHS = new Attribute("homeShot"); // Attributo numerico
        Attribute attAS = new Attribute("awayShot"); // Attributo numerico
        Attribute attHST = new Attribute("homeShotTarget"); // Attributo numerico
        Attribute attAST = new Attribute("awayShotTarget"); // Attributo numerico
        Attribute attHC = new Attribute("homeCorner"); // Attributo numerico
        Attribute attAC = new Attribute("awayCorner"); // Attributo numerico
        Attribute attHF = new Attribute("homeFouls"); // Attributo numerico
        Attribute attAF = new Attribute("awayFouls"); // Attributo numerico
        Attribute attHY = new Attribute("homeYellow"); // Attributo numerico
        Attribute attAY = new Attribute("awayYellow"); //Attributo numerico
        Attribute attHR = new Attribute("homeRed"); // Attributo numerico
        Attribute attAR = new Attribute("awayRed"); // Attributo numerico
        Attribute attDifferenceGHT = new Attribute("differenceGoalHT"); // Attributo numerico
        Attribute attPctHShot = new Attribute("pctHShot"); // Attributo numerico


        atts.add(attHomeTeam);
        atts.add(attAwayTeam);
        atts.add(attFTHG);
        atts.add(attFTAG);
        atts.add(attFTR);
        atts.add(attHTHG);
        atts.add(attHTAG);
        //atts.add(attHTR);
        atts.add(attHS);
        atts.add(attAS);
        atts.add(attHST);
        atts.add(attAST);
        atts.add(attHC);
        atts.add(attAC);
        atts.add(attHF);
        atts.add(attAF);
        atts.add(attHY);
        atts.add(attAY);
        atts.add(attHR);
        atts.add(attAR);
        atts.add(attDifferenceGHT);
        atts.add(attPctHShot);

        // Creiamo il dataset (classe Instances)
        Instances data = new Instances("SerieA", atts, 0);
        System.out.println("Dataset contains: " + data.numInstances() + " instances");

        // Popoliamo il dataset
        String rootDir = "./src/resources/";
        String fileName = "SerieA2016.csv";
        List<String> lines = Files.readAllLines(FileSystems.getDefault().getPath(rootDir, fileName));
        boolean first = true;
        for (String l : lines) {
            if(first) {
                first = false;
                continue;
            }
            String[] fields = l.split(",");
            // Creiamo una istanza (osservazione) da aggiungere al dataset
            Instance observation = new DenseInstance(21);
            observation.setValue(attHomeTeam, fields[2]);
            observation.setValue(attAwayTeam, fields[3]);
            observation.setValue(attFTHG, Double.parseDouble(fields[4]));
            observation.setValue(attFTAG, Double.parseDouble(fields[5]));
            observation.setValue(attFTR, fields[6].equals("H") || fields[6].equals("D") ? "H" : "A");
            observation.setValue(attHTHG, fields[7].equals("") ? 0 :  Double.parseDouble(fields[7]));
            observation.setValue(attHTAG, fields[8].equals("") ? 0 :  Double.parseDouble(fields[8]));
            //observation.setValue(attHTR, fields[9]);
            observation.setValue(attHS, Double.parseDouble(fields[10]));
            observation.setValue(attAS, Double.parseDouble(fields[11]));
            observation.setValue(attHST, Double.parseDouble(fields[12]));
            observation.setValue(attAST, Double.parseDouble(fields[13]));
            observation.setValue(attHC, Double.parseDouble(fields[16]));
            observation.setValue(attAC, Double.parseDouble(fields[17]));
            observation.setValue(attHF, Double.parseDouble(fields[14]));
            observation.setValue(attAF, Double.parseDouble(fields[15]));
            observation.setValue(attHY, Double.parseDouble(fields[18]));
            observation.setValue(attAY, Double.parseDouble(fields[19]));
            observation.setValue(attHR, Double.parseDouble(fields[20]));
            observation.setValue(attAR, Double.parseDouble(fields[21]));
            observation.setValue(attDifferenceGHT, (fields[7].equals("") ? 0 :  Double.parseDouble(fields[7])) - (fields[8].equals("") ? 0 :  Double.parseDouble(fields[8])));
            double pctHShot = Double.parseDouble(fields[10])/(Double.parseDouble(fields[10]) + Double.parseDouble(fields[11]));
            observation.setValue(attPctHShot, pctHShot);

            observation.setDataset(data);
            data.add(observation);
            if (!data.checkInstance(observation))
                throw new RuntimeException();
        }
        System.out.println("Dataset contains: " + data.numInstances() + " instances");

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(rootDir, "SerieA2016.arff"));
        saver.writeBatch();
    }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------//
////////////////////////////////////////////////////////////////ANALISI ESPLORATIVA////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------------------------------------------------------------------------------------//

    private static void generateShotVsGoals() {
        HashMap<String, Double> homeTeamShot = findDataDouble(2, 10);
        Map<String, Double> sortedMapHomeS = new TreeMap<>(homeTeamShot);
        HashMap<String, Double> awayTeamShot = findDataDouble(3, 11);
        Map<String, Double> sortedMapAwayS = new TreeMap<>(awayTeamShot);
        HashMap<String, Double> homeTeamGoals = findDataDouble(2, 4);
        Map<String, Double> sortedMapHomeG = new TreeMap<>(homeTeamGoals);
        HashMap<String, Double> awayTeamGoals = findDataDouble(3, 5);
        Map<String, Double> sortedMapAwayG = new TreeMap<>(awayTeamGoals);

        Map<String, Double> sortedMapHomeAwayS = sumDouble(sortedMapHomeS, sortedMapAwayS);
        Map<String, Double> sortedMapHomeAwayG = sumDouble(sortedMapHomeG, sortedMapAwayG);

        Map<String, Double> sortedMapHomeAwayPerc = relationDouble(sortedMapHomeAwayS, sortedMapHomeAwayG);

        makeGroupedBarChartDouble(sortedMapHomeAwayPerc, "Team", "Shots/Goals");
    }

    private static void generateFaulsVsYellows() {
        HashMap<String, Double> homeTeamFauls = findDataDouble(2, 14);
        Map<String, Double> sortedMapHomeF = new TreeMap<>(homeTeamFauls);
        HashMap<String, Double> awayTeamFauls = findDataDouble(3, 15);
        Map<String, Double> sortedMapAwayF = new TreeMap<>(awayTeamFauls);
        HashMap<String, Double> homeTeamYellow = findDataDouble(2, 18);
        Map<String, Double> sortedMapHomeY = new TreeMap<>(homeTeamYellow);
        HashMap<String, Double> awayTeamYellow = findDataDouble(3, 19);
        Map<String, Double> sortedMapAwayY = new TreeMap<>(awayTeamYellow);

        Map<String, Double> sortedMapHomeAwayF = sumDouble(sortedMapHomeF, sortedMapAwayF);
        Map<String, Double> sortedMapHomeAwayY = sumDouble(sortedMapHomeY, sortedMapAwayY);

        Map<String, Double> sortedMapHomeAwayPerc = relationDouble(sortedMapHomeAwayF, sortedMapHomeAwayY);

        makeGroupedBarChartDouble(sortedMapHomeAwayPerc, "Team", "Fauls/Yellow cards");
    }

    private static Map<String,Integer> relation(Map<String, Integer> homeAndAwayTeamAttr1, Map<String, Integer> homeAndAwayTeamAttr2) {
        Map<String, Integer> relation = new TreeMap<>();

        for (Map.Entry<String, Integer> entry: homeAndAwayTeamAttr1.entrySet()) {
            if (relation.containsKey(entry.getKey())) {
                int oldNumber = relation.get(entry.getKey());
                int newNumber = entry.getValue();
                relation.replace(entry.getKey(), oldNumber + newNumber);
            } else {
                relation.put(entry.getKey(), entry.getValue());
            }
        }

        for (Map.Entry<String, Integer> entry: homeAndAwayTeamAttr2.entrySet()) {
            if (relation.containsKey(entry.getKey())) {
                int oldNumber = relation.get(entry.getKey());
                int newNumber = entry.getValue();
                relation.replace(entry.getKey(), oldNumber / newNumber);
            } else {
                relation.put(entry.getKey(), entry.getValue());
            }
        }
        return relation;
    }

    private static Map<String,Double> relationDouble(Map<String, Double> homeAndAwayTeamAttr1, Map<String, Double> homeAndAwayTeamAttr2) {
        Map<String, Double> relation = new TreeMap<>();

        for (Map.Entry<String, Double> entry: homeAndAwayTeamAttr1.entrySet()) {
            if (relation.containsKey(entry.getKey())) {
                double oldNumber = relation.get(entry.getKey());
                double newNumber = entry.getValue();
                relation.replace(entry.getKey(), oldNumber + newNumber);
            } else {
                relation.put(entry.getKey(), entry.getValue());
            }
        }

        for (Map.Entry<String, Double> entry: homeAndAwayTeamAttr2.entrySet()) {
            if (relation.containsKey(entry.getKey())) {
                double oldNumber = relation.get(entry.getKey());
                double newNumber = entry.getValue();
                relation.replace(entry.getKey(), oldNumber / newNumber);
            } else {
                relation.put(entry.getKey(), entry.getValue());
            }
        }
        return relation;
    }

    private static Map<String,Integer> sum(Map<String, Integer> homeTeamAttr, Map<String, Integer> awayTeamAttr) {
        Map<String, Integer> sum = new TreeMap<>();

        for (Map.Entry<String, Integer> entry: homeTeamAttr.entrySet()) {
            if (sum.containsKey(entry.getKey())) {
                int oldNumber = sum.get(entry.getKey());
                int newNumber = entry.getValue();
                sum.replace(entry.getKey(), oldNumber + newNumber);
            } else {
                sum.put(entry.getKey(), entry.getValue());
            }
        }

        for (Map.Entry<String, Integer> entry: awayTeamAttr.entrySet()) {
            if (sum.containsKey(entry.getKey())) {
                int oldNumber = sum.get(entry.getKey());
                int newNumber = entry.getValue();
                sum.replace(entry.getKey(), oldNumber + newNumber);
            } else {
                sum.put(entry.getKey(), entry.getValue());
            }
        }
        return sum;
    }
    private static Map<String,Double> sumDouble(Map<String, Double> homeTeamAttr, Map<String, Double> awayTeamAttr) {
        Map<String, Double> sum = new TreeMap<>();

        for (Map.Entry<String, Double> entry: homeTeamAttr.entrySet()) {
            if (sum.containsKey(entry.getKey())) {
                double oldNumber = sum.get(entry.getKey());
                double newNumber = entry.getValue();
                sum.replace(entry.getKey(), oldNumber + newNumber);
            } else {
                sum.put(entry.getKey(), entry.getValue());
            }
        }

        for (Map.Entry<String, Double> entry: awayTeamAttr.entrySet()) {
            if (sum.containsKey(entry.getKey())) {
                double oldNumber = sum.get(entry.getKey());
                double newNumber = entry.getValue();
                sum.replace(entry.getKey(), oldNumber + newNumber);
            } else {
                sum.put(entry.getKey(), entry.getValue());
            }
        }
        return sum;
    }
    private static void generateFaulsVsGoals() {
        ArrayList<Integer> fauls = new ArrayList<>();
        ArrayList<Integer> goals = new ArrayList<>();
        for (CSVRecord record : completeList) {
            //falli fatti da away
            fauls.add(Integer.parseInt(record.get(15)));
            //goal subiti da away
            goals.add(Integer.parseInt(record.get(4)));
            //falli fatti da home
            fauls.add(Integer.parseInt(record.get(14)));
            //goal subiti da home
            goals.add(Integer.parseInt(record.get(5)));
        }

        makeIntScatterPlot(fauls, goals, "Fauls", "Goals");
    }

    private static void generateGoalsAtHomeVsGoalsAway() {
        HashMap<String, Integer> homeTeamAndGoalsHome = findData(2, 4);
        Map<String, Integer> sortedMapHome = new TreeMap<>(homeTeamAndGoalsHome);
        HashMap<String, Integer> awayTeamAndGoalsAway = findData(3, 5);
        Map<String, Integer> sortedMapAway = new TreeMap<>(awayTeamAndGoalsAway);

        //makeGroupedBarChart(sortedMapHome, sortedMapAway);
        makeScatterPlot(sortedMapHome, sortedMapAway, "Goals at home", "Goals away");
    }

    //trova valori per ogni team
    private static HashMap<String, Integer> findData(int team, int wantedValue) {
        HashMap<String, Integer> homeTeamAndGoalsHome = new HashMap<>();
        for (CSVRecord record : completeList) {
            if (homeTeamAndGoalsHome.containsKey(record.get(team))) {
                int oldNumber = homeTeamAndGoalsHome.get(record.get(team));
                int newNumber = Integer.parseInt(record.get(wantedValue));
                homeTeamAndGoalsHome.replace(record.get(team), oldNumber + newNumber);

            } else {
                homeTeamAndGoalsHome.put(record.get(team), Integer.parseInt(record.get(wantedValue)));
            }
        }
        return homeTeamAndGoalsHome;
    }

    //trova valori per ogni team
    private static HashMap<String, Double> findDataDouble(int team, int wantedValue) {
        HashMap<String, Double> homeTeamAndGoalsHome = new HashMap<>();
        for (CSVRecord record : completeList) {
            if (homeTeamAndGoalsHome.containsKey(record.get(team))) {
                double oldNumber = homeTeamAndGoalsHome.get(record.get(team));
                double newNumber = Integer.parseInt(record.get(wantedValue));
                homeTeamAndGoalsHome.replace(record.get(team), oldNumber + newNumber);

            } else {
                homeTeamAndGoalsHome.put(record.get(team), Double.parseDouble(record.get(wantedValue)));
            }
        }
        return homeTeamAndGoalsHome;
    }
    private static void generateGoalsVsShot() {
        ArrayList<Integer> shotsOnTarget = new ArrayList<>();
        ArrayList<Integer> goals = new ArrayList<>();
        for (CSVRecord record : completeList) {
            shotsOnTarget.add(Integer.parseInt(record.get(12)));
            goals.add(Integer.parseInt(record.get(4)));
            shotsOnTarget.add(Integer.parseInt(record.get(13)));
            goals.add(Integer.parseInt(record.get(5)));
        }

        //calcola intercetta correlazione e media
        ArrayList<Attribute> atts = new ArrayList<>();
        Attribute attGoals = new Attribute("goals"); // Attributo numerico
        Attribute attShot = new Attribute("shot on target"); // Attributo numerico
        atts.add(attGoals);
        atts.add(attShot);

        // Creiamo il dataset (classe Instances)
        Instances data = new Instances("SerieA", atts, 0);
        System.out.println("Dataset contains: " + data.numInstances() + " instances");
        int i = 0;
        for (Integer value : goals) {
            Instance observation = new DenseInstance(2);

            observation.setValue(attGoals, value);
            observation.setValue(attShot, shotsOnTarget.get(i));

            observation.setDataset(data);
            data.add(observation);
            if (!data.checkInstance(observation))
                throw new RuntimeException();
            i++;
        }
        System.out.println("Dataset contains: " + data.numInstances() + " instances");

        makeScatterPlot(shotsOnTarget, goals,data ,attShot, generateStatistics(data, attGoals), "Shots on target", "Goals");
    }

    private static SimpleLinearRegression generateStatistics(Instances data, Attribute attY){
        System.out.println("Calcolo la regressione");
        data.setClass(attY);
        SimpleLinearRegression lr = new SimpleLinearRegression();
        try {
            lr.buildClassifier(data); // Calcola la regressione
            System.out.println("Intercept (b): " + lr.getIntercept()); // b
            System.out.println("Slope (w):" + lr.getSlope()); // w
            CorrelationAttributeEval cae = new CorrelationAttributeEval();
            cae.buildEvaluator(data); // prepara
            System.out.println("Correlation coefficient: " + cae.evaluateAttribute(1));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return lr;
    }

    //legge file con libreria prof per plot di base
    private static ArrayList<CSVRecord> readFile() throws IOException {
        Reader in = new FileReader("./src/resources/SerieA2016.csv");
        Iterable<CSVRecord> records = CSVFormat.RFC4180.parse(in);
        ArrayList<CSVRecord> completeTimeList = new ArrayList<>();

        int counter = 0;
        for (CSVRecord record : records) {
            if (counter == 0) {
                System.out.println("----------------------ATTRIBUTES----------------------");
                for (int i = 0; i < record.size(); i++) {
                    System.out.print(i + "-" + record.get(i) + "    ");
                }
                System.out.println();
            } else {
                completeTimeList.add(record);
            }
            counter++;
        }

        System.out.println("Number of records: " + counter);
        return completeTimeList;
    }


//-------------------------------------------------------------------------------------------------------------------------------------------------------------//
///////////////////////////////////////////////////GRAFICI, PLOT, ISTOGRAMMI E ROC/////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------------------------------------------------------------------------------------//

    //crea scatter dei tiri e dei goals per ogni match con retta regressione
    private static void makeScatterPlot(ArrayList<Integer> xList, ArrayList<Integer> yList, Instances data, Attribute attShot, SimpleLinearRegression lr, String xString, String yString) {
        ArrayList<Double> xFinal = new ArrayList<>();
        ArrayList<Double> yFinal = new ArrayList<>();
        ArrayList<Double> x2List = new ArrayList<Double>();
        x2List.add(data.kthSmallestValue(attShot, 1));
        x2List.add(data.kthSmallestValue(attShot, data.numInstances()));

        ArrayList<Double> y2List = new ArrayList<Double>();
        for (double x : x2List)
            y2List.add(lr.getIntercept() + lr.getSlope() * x);


        for (Integer value : xList) {
            xFinal.add(value +  Math.random());
        }
        for (Integer value : yList) {
            yFinal.add(value +  Math.random());
        }

        JSONObject marker = new JSONObject();
        marker.put("size", 10);
        marker.put("color", "rgba(255, 0, 0, .4)");

        JSONObject trace = new JSONObject();
        trace.put("name", "Match");
        trace.put("x", xFinal);
        trace.put("y", yFinal);
        trace.put("type", "scatter");
        trace.put("mode", "markers");
        trace.put("marker", marker);

        JSONObject trace2 = new JSONObject();
        trace2.put("name", "Regression");
        trace2.put("x", x2List);
        trace2.put("y", y2List);
        trace2.put("type", "scatter");
        trace2.put("mode", "lines");

        JSONArray plotData = new JSONArray();
        plotData.add(trace);
        plotData.add(trace2);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", xString);

        JSONObject yaxis = new JSONObject();
        yaxis.put("title", yString);

        JSONObject layout = new JSONObject();
        layout.put("title", yString + " vs " + xString);
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        JSONObject figure = new JSONObject();
        figure.put("data", plotData);
        figure.put("layout", layout);

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    //crea curva ROC
    private static void makeROC(ArrayList<ArrayList<Double>> xList, ArrayList<ArrayList<Double>> yList, String xString, String yString) {
        /*
        Map<Double, Double> sortedList = new TreeMap<>();
        int i = 0;
        for (Double xVal : xList) {
            sortedList.put(xVal, yList.get(i));
            i++;
        }*/

        JSONObject marker = new JSONObject();
        marker.put("size", 10);
        marker.put("color", "rgba(255, 0, 0, .4)");

        JSONObject trace0 = new JSONObject();
        trace0.put("name", "2 attr.");
        trace0.put("x", xList.get(0));
        trace0.put("y", yList.get(0));
        trace0.put("type", "scatter");
        trace0.put("mode", "lines");

        JSONObject trace1 = new JSONObject();
        trace1.put("name", "Difference goals, HT");
        trace1.put("x", xList.get(1));
        trace1.put("y", yList.get(1));
        trace1.put("type", "scatter");
        trace1.put("mode", "lines");

        JSONObject trace2 = new JSONObject();
        trace2.put("name", "%Shot Home Team, FT");
        trace2.put("x", xList.get(2));
        trace2.put("y", yList.get(2));
        trace2.put("type", "scatter");
        trace2.put("mode", "lines");
/*
        JSONObject trace3 = new JSONObject();
        trace3.put("name", "RandomForest");
        trace3.put("x", xList.get(3));
        trace3.put("y", yList.get(3));
        trace3.put("type", "scatter");
        trace3.put("mode", "lines");
*/
        List<Double> xDummy = new ArrayList<>();
        xDummy.add(0.);
        xDummy.add(1.);
        JSONObject trace4 = new JSONObject();
        trace4.put("name", "Dummy");
        trace4.put("x", xDummy);
        trace4.put("y", xDummy);
        trace4.put("type", "scatter");
        trace4.put("mode", "lines");



        JSONArray plotData = new JSONArray();
        plotData.add(trace0);
        plotData.add(trace1);
        plotData.add(trace2);
        //plotData.add(trace3);
        plotData.add(trace4);

        JSONArray range = new JSONArray();
        range.add(0);
        range.add(1);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", xString);
        xaxis.put("range", range);
        JSONObject yaxis = new JSONObject();
        yaxis.put("title", yString);
        yaxis.put("range", range);

        JSONObject layout = new JSONObject();
        layout.put("title", xString + " vs " + yString);
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        JSONObject figure = new JSONObject();
        figure.put("data", plotData);
        figure.put("layout", layout);

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    //crea scatter date due mappe
    private static void makeScatterPlot(Map<String, Integer> sortedMapHome, Map<String, Integer> sortedMapAway, String xString, String yString) {
        ArrayList<Double> xFinal = new ArrayList<>();
        ArrayList<Double> yFinal = new ArrayList<>();
        ArrayList<String> textList = new ArrayList<>();
        ArrayList<Double> x2List = new ArrayList<>();
        ArrayList<Double> y2List = new ArrayList<>();

        for (Map.Entry<String, Integer> entry : sortedMapHome.entrySet()) {
            xFinal.add(entry.getValue() +  Math.random());
            textList.add(entry.getKey());
        }
        for (Map.Entry<String, Integer> entry : sortedMapAway.entrySet()) {
            yFinal.add(entry.getValue() +  Math.random());
        }

        x2List.add(0.);
        y2List.add(0.);
        x2List.add(50.);
        y2List.add(50.);


        JSONObject marker = new JSONObject();
        marker.put("size", 10);
        marker.put("color", "rgba(0, 0, 255, 1)");

        JSONObject trace = new JSONObject();
        trace.put("name", "Team");
        trace.put("x", xFinal);
        trace.put("y", yFinal);
        trace.put("text", textList);
        trace.put("type", "scatter");
        trace.put("mode", "markers");
        trace.put("marker", marker);

        JSONObject trace2 = new JSONObject();
        trace2.put("name", "Goals away = goals at home");
        trace2.put("x", x2List);
        trace2.put("y", y2List);
        trace2.put("type", "scatter");
        trace2.put("mode", "lines");

        JSONArray plotData = new JSONArray();
        plotData.add(trace);
        plotData.add(trace2);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", xString);

        JSONObject yaxis = new JSONObject();
        yaxis.put("title", yString);

        JSONObject layout = new JSONObject();
        layout.put("title", yString + " vs " + xString);
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        JSONObject figure = new JSONObject();
        figure.put("data", plotData);
        figure.put("layout", layout);

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //crea scatter per due liste con interi
    private static void makeIntScatterPlot(ArrayList<Integer> xList, ArrayList<Integer> yList, String xString, String yString) {
        ArrayList<Double> xFinal = new ArrayList<>();
        ArrayList<Double> yFinal = new ArrayList<>();

        for (Integer value : xList) {
            xFinal.add(value +  Math.random());
        }
        for (Integer value : yList) {
            yFinal.add(value +  Math.random());
        }

        JSONObject marker = new JSONObject();
        marker.put("size", 7);
        marker.put("color", "rgba(255, 0, 0, .4)");

        JSONObject trace = new JSONObject();
        trace.put("name", "Item");
        trace.put("x", xFinal);
        trace.put("y", yFinal);
        trace.put("type", "scatter");
        trace.put("mode", "markers");
        trace.put("marker", marker);

        JSONArray plotData = new JSONArray();
        plotData.add(trace);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", xString);

        JSONObject yaxis = new JSONObject();
        yaxis.put("title", yString);

        JSONObject layout = new JSONObject();
        layout.put("title", xString + " vs " + yString);
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        JSONObject figure = new JSONObject();
        figure.put("data", plotData);
        figure.put("layout", layout);

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //crea scatter per due liste per double
    private static void makeDoubleScatterPlot(ArrayList<Double> xList, ArrayList<Double> yList, String xString, String yString) {
        ArrayList<Double> xFinal = new ArrayList<>();
        ArrayList<Double> yFinal = new ArrayList<>();

        for (Double value : xList) {
            xFinal.add(value);
        }
        for (Double value : yList) {
            yFinal.add(value);
        }

        JSONObject marker = new JSONObject();
        marker.put("size", 7);
        marker.put("color", "rgba(255, 0, 0, .4)");

        JSONObject trace = new JSONObject();
        trace.put("name", "Item");
        trace.put("x", xFinal);
        trace.put("y", yFinal);
        trace.put("type", "scatter");
        trace.put("mode", "markers");
        trace.put("marker", marker);

        JSONArray plotData = new JSONArray();
        plotData.add(trace);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", xString);

        JSONObject yaxis = new JSONObject();
        yaxis.put("title", yString);

        JSONObject layout = new JSONObject();
        layout.put("title", yString + " vs " + xString);
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        JSONObject figure = new JSONObject();
        figure.put("data", plotData);
        figure.put("layout", layout);

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //crea 2 istogrammi per squadra
    private static void makeGroupedBarChart(Map<String, Integer> sortedMapHome, Map<String, Integer> sortedMapAway) {

        ArrayList<String> xAxis = new ArrayList<>();

        xAxis.addAll(sortedMapHome.keySet());

        JSONObject trace0 = new JSONObject();
        trace0.put("x", xAxis);
        trace0.put("y", sortedMapHome.values());
        trace0.put("name", "goals at home");
        trace0.put("type", "bar");

        JSONObject trace1 = new JSONObject();
        trace1.put("x", xAxis);
        trace1.put("y", sortedMapAway.values());
        trace1.put("name", "goals away");
        trace1.put("type", "bar");

        JSONArray data = new JSONArray();
        data.add(trace0);
        data.add(trace1);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", "Teams");

        JSONObject yaxis = new JSONObject();
        yaxis.put("title", "Goals made");


        JSONObject layout = new JSONObject();
        layout.put("title", "Goals at Home / Goals Away");
        layout.put("barmode", "group");
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        JSONObject figure = new JSONObject();
        figure.put("data", data);
        figure.put("layout", layout);
        System.out.println("json");

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    //crea istogramma per squadra
    private static void makeGroupedBarChart(Map<String, Integer> sortedMapHome, String xString, String yString) {

        ArrayList<String> xAxis = new ArrayList<>();

        xAxis.addAll(sortedMapHome.keySet());

        JSONObject trace0 = new JSONObject();
        trace0.put("x", xAxis);
        trace0.put("y", sortedMapHome.values());
        trace0.put("name", "goals at home");
        trace0.put("type", "bar");


        JSONArray data = new JSONArray();
        data.add(trace0);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", xString);

        JSONObject yaxis = new JSONObject();
        yaxis.put("title", yString);

        JSONObject layout = new JSONObject();
        layout.put("title", xString + " vs " + yString);
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        //layout.put("barmode", "group");

        JSONObject figure = new JSONObject();
        figure.put("data", data);
        figure.put("layout", layout);
        System.out.println("json");

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    //crea istogramma per squadra
    private static void makeGroupedBarChartDouble(Map<String, Double> sortedMapHome, String xString, String yString) {
        ArrayList<String> xAxis = new ArrayList<>();

        Map<Double, String> ascendingValue = new TreeMap<>();
        for (Map.Entry<String, Double> entry: sortedMapHome.entrySet()) {
            ascendingValue.put(entry.getValue(), entry.getKey());
        }
        xAxis.addAll(ascendingValue.values());

        JSONObject trace0 = new JSONObject();
        trace0.put("x", xAxis);
        trace0.put("y",  ascendingValue.keySet());
        trace0.put("name", "goals at home");
        trace0.put("type", "bar");


        JSONArray data = new JSONArray();
        data.add(trace0);

        JSONObject xaxis = new JSONObject();
        xaxis.put("title", xString);

        JSONObject yaxis = new JSONObject();
        yaxis.put("title", yString);

        JSONObject layout = new JSONObject();
        layout.put("title", yString + " vs " + xString);
        layout.put("xaxis", xaxis);
        layout.put("yaxis", yaxis);

        //layout.put("barmode", "group");

        JSONObject figure = new JSONObject();
        figure.put("data", data);
        figure.put("layout", layout);
        System.out.println("json");

        try (FileWriter file = new FileWriter("./src/resources/plot.json")) {
            file.write(figure.toJSONString());
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
