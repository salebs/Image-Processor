import java.awt.LayoutManager;
import java.awt.event.*;
import javax.swing.*;


/**
 * This class handles the information of the process' thread. Enables the user to see the progress of the prcoess and cancel the process if wanted.
 * 
 * @author Benjamin Sale
 */
class GuiWindow extends JFrame{

    /**
     * Create a window for every time the process button is hit. Includes a way for the user to cancel the process.
     */
    public GuiWindow(Process process, JButton button) {    
        setTitle("Process Status");
        setSize(400, 200);
        setLayout((LayoutManager)null);
        setLocation(950, 100);

        JButton cancel = new JButton("CANCEL");
        cancel.setBounds(150, 100, 100, 25);
        cancel.setActionCommand("cancel");
        cancel.addActionListener(new ActionListener() { 
            public void actionPerformed(ActionEvent e) { 
                process.destroy();
                button.setEnabled(true);
                dispose();
            } 
        });
        add(cancel);
    }
}
