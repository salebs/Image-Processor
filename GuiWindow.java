import java.awt.LayoutManager;
import java.awt.event.*;
import javax.swing.*;

class GuiWindow extends JFrame{

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
