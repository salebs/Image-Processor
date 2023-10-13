import java.awt.LayoutManager;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

public class GuiMain extends JFrame implements ActionListener {

    private static JButton[] buttons;
    private static JTextField[] fields;

    public void actionPerformed(ActionEvent event) {
        
    }

    public GuiMain() {
        buttons = new JButton[1];
        fields = new JTextField[9];

        setTitle("Image Processor");
        setSize(400, 350);
        setLayout((LayoutManager)null);
        setLocation(500, 100);

        JLabel introduction = new JLabel("This is the introduction. Enter information below.");
        introduction.setBounds(20, 0, 300, 20);
        add(introduction);
        
        JLabel directory = new JLabel("Enter directory:");
        directory.setBounds(20, 25, 100, 20);
        add(directory);
        JTextField directoryResponse = new JTextField("");
        directoryResponse.setBounds(145, 25, 200, 20);
        directoryResponse.setName("directory");
        add(directoryResponse);

        JLabel modelType = new JLabel("Enter model type:");
        modelType.setBounds(20, 50, 150, 20);
        add(modelType);
        JTextField modelTypeResponse = new JTextField("");
        modelTypeResponse.setBounds(145, 50, 200, 20);
        modelTypeResponse.setName("model type");
        add(modelTypeResponse);

        JLabel negativePatchCount = new JLabel("Enter negative patch count:");
        negativePatchCount.setBounds(20, 75, 150, 20);
        add(negativePatchCount);
        JTextField negativePatchCountResponse = new JTextField("");
        negativePatchCountResponse.setBounds(145, 75, 200, 20);
        negativePatchCountResponse.setName("negative patch number");
        add(negativePatchCountResponse);

        JLabel imageSize = new JLabel("Enter image size:");
        imageSize.setBounds(20, 100, 150, 20);
        add(imageSize);
        JTextField imageSizeResponse = new JTextField("");
        imageSizeResponse.setBounds(145, 100, 100, 20);
        imageSizeResponse.setName("image size");
        add(imageSizeResponse);

        JLabel patchSize = new JLabel("Enter patch size:");
        patchSize.setBounds(20, 125, 150, 20);
        add(patchSize);
        JTextField patchSizeResponse = new JTextField("");
        patchSizeResponse.setBounds(145, 125, 100, 20);
        patchSizeResponse.setName("patch size");
        add(patchSizeResponse);

        JLabel cellSize = new JLabel("Enter cell size:");
        cellSize.setBounds(20, 150, 150, 20);
        add(cellSize);
        JTextField cellSizeResponse = new JTextField("");
        cellSizeResponse.setBounds(145, 150, 100, 20);
        cellSizeResponse.setName("cell size");
        add(cellSizeResponse);

        JLabel blockSize = new JLabel("Enter block size:");
        blockSize.setBounds(20, 175, 150, 20);
        add(blockSize);
        JTextField blockSizeResponse = new JTextField("");
        blockSizeResponse.setBounds(145, 175, 100, 20);
        blockSizeResponse.setName("block size");
        add(blockSizeResponse);

        JLabel stepSize = new JLabel("Enter step size:");
        stepSize.setBounds(20, 200, 150, 20);
        add(stepSize);
        JTextField stepSizeResponse = new JTextField("");
        stepSizeResponse.setBounds(145, 200, 100, 20);
        stepSizeResponse.setName("step size");
        add(stepSizeResponse);

        JLabel scales = new JLabel("Enter scales:");
        scales.setBounds(20, 225, 150, 20);
        add(scales);
        JTextField scalesResponse = new JTextField("");
        scalesResponse.setBounds(145, 225, 100, 20);
        scalesResponse.setName("scales");
        add(scalesResponse);

        JCheckBox display = new JCheckBox("display images"); 
        display.setBounds(20, 250, 150, 20);
        add(display);

        JButton process = new JButton("PROCESS");
        process.setBounds(140, 300, 100, 25);
        process.setName("process");
        //process.setActionCommand("process images");
        //process.setActionListener(this);
        add(process);
        buttons[0] = process;
    }

    public static void main(String[] args) {
        GuiMain window = new GuiMain();
        window.setDefaultCloseOperation(3);
        window.setVisible(true);
     }
}