import java.awt.LayoutManager;
import java.awt.event.*;
import javax.swing.*;

import java.io.IOException;

/**
 * This class enables the user to easily access the Image Processor with specified parameters and without touching any code.
 * 
 * @author Benjamin Sale
 */
public class GuiMain extends JFrame implements ActionListener{

    /**
     * The fields will have the user-defined arguments for the terminal command.
     */
    private static JTextField[] fields;

    /**
     * The view will have the flags to determine if the user wants to see the trained or testing images.
     */
    private static JCheckBox[] view;

    /**
     * The button is the process button.
     */
    private JButton button;

    /**
     * When the process button is hit, a process thread is spun off to construct an Image Processor with the user specified paramters. It will also disable the process button and create a GuiWindow.
     * 
     * @param args a string array with the appropriate terminal command arguments
     */
    public void run(String[] args) {
        try {
            String command = "C:\\Users\\benss\\AppData\\Local\\Programs\\Python\\Python39\\python.exe " + args[0] + "\\Process.py";
            for (String arg : args) { command += " " + arg; }
            System.out.println(command);
            button.setEnabled(false);
            ProcessBuilder processBuilder = new ProcessBuilder(command.split(" "));
            Process process = processBuilder.start();
            GuiWindow status = new GuiWindow(process, button);
            status.setVisible(true);
            return;
        } catch (IOException e) {
            System.out.println("Exception is caught");
            e.printStackTrace();
            return;
        }
    }

    /**
     * Ensure all fields are filled out in GUI and starts the process based off the user-defined parameters.
     */
    public void actionPerformed(ActionEvent event) {
        String[] args = new String[16];
        for (int i=0; i<fields.length; i++) {
            String fieldText = fields[i].getText();
            if (fieldText.equals("")) { 
                JOptionPane.showMessageDialog(null, "Enter information for all fields.", "Warning", JOptionPane.WARNING_MESSAGE);
                return;
            }
            args[i] = fieldText;
        }
        for (int i=0; i<view.length; i++) {
            boolean check = view[i].isSelected();
            String boxString;
            if (check) { boxString = "true"; }
            else { boxString = "false"; }
            args[fields.length + i] = boxString;
        }
        run(args);
    }
    
    /**
     * Create a GUI and install all appropriate fields and check boxes to gather terminal command arguments, as well as process button.
     */
    public GuiMain() {
        fields = new JTextField[14];
        view = new JCheckBox[2];

        setTitle("Image Processor");
        setSize(400, 400);
        setLayout((LayoutManager)null);
        setLocation(500, 100);

        JLabel introduction = new JLabel("This is the introduction. Enter information below.");
        introduction.setBounds(20, 0, 300, 20);
        add(introduction);
        
        JLabel directory = new JLabel("Enter directory:");
        directory.setBounds(20, 25, 100, 20);
        add(directory);
        JTextField directoryResponse = new JTextField("C:\\Users\\benss\\PycharmProjects\\COSC-470\\OCT\\Image-Processor");
        directoryResponse.setBounds(160, 25, 200, 20);
        directoryResponse.setName("directory");
        add(directoryResponse);
        fields[0] = directoryResponse;

        JLabel modelType = new JLabel("Enter model type:");
        modelType.setBounds(20, 50, 150, 20);
        add(modelType);
        JTextField modelTypeResponse = new JTextField("KNeighborsClassifier");
        modelTypeResponse.setBounds(160, 50, 200, 20);
        modelTypeResponse.setName("model type");
        add(modelTypeResponse);
        fields[1] = modelTypeResponse;

        JLabel negativePatchCount = new JLabel("Enter negative amount:");
        negativePatchCount.setBounds(20, 75, 175, 20);
        add(negativePatchCount);
        JTextField negativePatchCountResponse = new JTextField("100");
        negativePatchCountResponse.setBounds(160, 75, 25, 20);
        negativePatchCountResponse.setName("negative patch number");
        add(negativePatchCountResponse);
        fields[2] = negativePatchCountResponse;

        JLabel imageSize = new JLabel("Enter image size:");
        imageSize.setBounds(20, 100, 150, 20);
        add(imageSize);
        JTextField imageSizeMinResponse = new JTextField("1000");
        imageSizeMinResponse.setBounds(160, 100, 30, 20);
        imageSizeMinResponse.setName("image size minimum");
        add(imageSizeMinResponse);
        fields[3] = imageSizeMinResponse;
        JTextField imageSizeMaxResponse = new JTextField("500");
        imageSizeMaxResponse.setBounds(200, 100, 30, 20);
        imageSizeMaxResponse.setName("image size maximum");
        add(imageSizeMaxResponse);
        fields[4] = imageSizeMaxResponse;

        JLabel patchSize = new JLabel("Enter patch size:");
        patchSize.setBounds(20, 125, 150, 20);
        add(patchSize);
        JTextField patchSizeMinResponse = new JTextField("100");
        patchSizeMinResponse.setBounds(160, 125, 25, 20);
        patchSizeMinResponse.setName("patch size minimum");
        add(patchSizeMinResponse);
        fields[5] = patchSizeMinResponse;
        JTextField patchSizeMaxResponse = new JTextField("100");
        patchSizeMaxResponse.setBounds(190, 125, 25, 20);
        patchSizeMaxResponse.setName("patch size maximum");
        add(patchSizeMaxResponse);
        fields[6] = patchSizeMaxResponse;

        JLabel cellSize = new JLabel("Enter cell size:");
        cellSize.setBounds(20, 150, 150, 20);
        add(cellSize);
        JTextField cellSizeMinResponse = new JTextField("8");
        cellSizeMinResponse.setBounds(160, 150, 25, 20);
        cellSizeMinResponse.setName("cell size minimum");
        add(cellSizeMinResponse);
        fields[7] = cellSizeMinResponse;
        JTextField cellSizeMaxResponse = new JTextField("8");
        cellSizeMaxResponse.setBounds(190, 150, 25, 20);
        cellSizeMaxResponse.setName("cell size maximum");
        add(cellSizeMaxResponse);
        fields[8] = cellSizeMaxResponse;

        JLabel blockSize = new JLabel("Enter block size:");
        blockSize.setBounds(20, 175, 100, 20);
        add(blockSize);
        JTextField blockSizeMinResponse = new JTextField("1");
        blockSizeMinResponse.setBounds(160, 175, 25, 20);
        blockSizeMinResponse.setName("block size minimum");
        add(blockSizeMinResponse);
        fields[9] = blockSizeMinResponse;
        JTextField blockSizeMaxResponse = new JTextField("1");
        blockSizeMaxResponse.setBounds(190, 175, 25, 20);
        blockSizeMaxResponse.setName("block size maximum");
        add(blockSizeMaxResponse);
        fields[10] = blockSizeMaxResponse;

        JLabel stepSize = new JLabel("Enter step size:");
        stepSize.setBounds(20, 200, 150, 20);
        add(stepSize);
        JTextField stepSizeResponse = new JTextField("0.5");
        stepSizeResponse.setBounds(160, 200, 25, 20);
        stepSizeResponse.setName("step size");
        add(stepSizeResponse);
        fields[11] = stepSizeResponse;

        JLabel scales = new JLabel("Enter scales:");
        scales.setBounds(20, 225, 150, 20);
        add(scales);
        JTextField scalesMinResponse = new JTextField("0.5");
        scalesMinResponse.setBounds(160, 225, 25, 20);
        scalesMinResponse.setName("scales minimum");
        add(scalesMinResponse);
        fields[12] = scalesMinResponse;
        JTextField scalesMaxResponse = new JTextField("2");
        scalesMaxResponse.setBounds(190, 225, 25, 20);
        scalesMaxResponse.setName("scales maximum");
        add(scalesMaxResponse);
        fields[13] = scalesMaxResponse;

        JCheckBox displayTrain = new JCheckBox("display training images"); 
        displayTrain.setBounds(125, 250, 160, 20);
        add(displayTrain);
        view[0] = displayTrain;

        JCheckBox displayTest = new JCheckBox("display testing images"); 
        displayTest.setBounds(125, 275, 150, 20);
        add(displayTest);
        view[1] = displayTest;

        JButton process = new JButton("PROCESS");
        process.setBounds(150, 300, 100, 25);
        process.setActionCommand("process");
        process.addActionListener(this);
        add(process);
        button = process;
    }

    public static void main(String[] args) {
        GuiMain window = new GuiMain();
        window.setDefaultCloseOperation(3);
        window.setVisible(true);
     }
}
