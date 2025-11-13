package core;
import edu.princeton.cs.algs4.StdDraw;
import tileengine.TERenderer;
import tileengine.TETile;
import tileengine.Tileset;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

public class GenerateWorldFromInput {

    private World world;

    public GenerateWorldFromInput() {
    }
    public void displayMenu(String str) {
        while (true) {
            StdDraw.setCanvasSize(700, 700);
            StdDraw.setXscale(0, 700);
            StdDraw.setYscale(0, 700);
            Font font1 = new Font("Times New Roman", Font.ITALIC, 35);
            StdDraw.setFont(font1);
            StdDraw.clear(Color.black);
            StdDraw.setPenColor(Color.orange);
            StdDraw.text(300, 450, "Menu");
            StdDraw.text(300, 350, "N - New Game");
            StdDraw.text(300, 250, "L - Load Game");
            StdDraw.text(300, 150, "Q - Quit Game");
            StdDraw.show();
            if (str == null) {
                String input;
                while (!StdDraw.hasNextKeyTyped()) {
                    input = "";
                }
                char input0 = StdDraw.nextKeyTyped();
                input = String.valueOf(input0).toUpperCase(); //puts the input in upper case for processing
                switch (input) {
                    case "Q":
                        System.exit(0);
                        break;
                    case "L":
                        loadWorld(null);
                        return;
                    case "N":
                        seedGenerator(null);
                        return;
                    default:
                        StdDraw.text(300, 350, "Enter N, L, or Q!");
                        StdDraw.pause(2500);
                }
            } else { //passed in by autograder
                String input = String.valueOf(str.charAt(0)).toUpperCase();
                switch (input) {
                    case "N":
                        seedGenerator(str.substring(1));
                        return;
                    case "L":
                        loadWorld(str.substring(1));
                        return;
                    case "Q":
                        System.exit(0);
                        break;
                    default:
                        StdDraw.text(300, 350, "Enter N, L, or Q!");
                        StdDraw.pause(2500);
                }
            }
        }
    }

    public void loadWorld(String moves) {
        TETile[][] w = new TETile[30][30];
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 30; j++) {
                w[i][j] = Tileset.NOTHING;
            }
        }
        Scanner s;
        try {
            s = new Scanner(new File("savedWorld.txt"));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        for (int i = 0; i < w.length; i++) {
            String line = s.nextLine();
            for (int j = 0; j < line.length(); j++) {
                w[i][j] = Tileset.searchByChar(line.charAt(j));
            }
        }
        world = new World(w);
        world.setCharacter(s.nextInt(), s.nextInt());
        runGame(moves);
    }

    public void seedGenerator(String str) {
        if (str != null) {
            str = str.toLowerCase();
            long seedAsLong = Long.parseLong(str.substring(0, str.indexOf("s")));
            world = new World(30, 30, seedAsLong);
            runGame(str.substring(str.indexOf("s") + 1));
            return;
        }
        StdDraw.clear(Color.black);
        StdDraw.text(300, 350, "Enter a seed (number), followed by S");
        String seedAsString = ""; //stores the seed for visualization

        while (true) {
            if (StdDraw.hasNextKeyTyped()) {
                char c = StdDraw.nextKeyTyped();
                if (c == 's' || c == 'S') {
                    long seedAsLong = Long.parseLong(seedAsString);
                    world = new World(30, 30, seedAsLong); //generates the world with the seed
                    return;
                } else if (Character.isDigit(c)) { //this displays the seed while it's being typed
                    seedAsString += c;
                    StdDraw.clear(Color.black);
                    StdDraw.setPenColor(Color.orange);
                    StdDraw.text(300, 550, seedAsString);
                }
            }
        }
    }

    public void runGame(String moves) {
        if (moves != null) {
            for (int i = 0; i < moves.length(); i++) {
                char input = moves.charAt(i);
                if (input == 'W' || input == 'w') {
                    world.updateLocation(1, 0);
                } else if (input == 'A' || input == 'a') {
                    world.updateLocation(0, -1);
                } else if (input == 'S' || input == 's') {
                    world.updateLocation(-1, 0);
                } else if (input == 'D' || input == 'd') {
                    world.updateLocation(0, 1);
                } else if (input == ':' && i < moves.length() - 1) {
                    input = moves.charAt(i + 1);
                    if (input == 'q' || input == 'Q') {
                        saveWorld();
                        return;
                    }
                }
            }
            return;
        }
        TERenderer ter = new TERenderer();
        ter.initialize(30, 30);
        ter.renderFrame(world.getWorld());
        boolean colon = false;
        while (true) {

            char input = ' ';
            if (StdDraw.hasNextKeyTyped()) {
                input = StdDraw.nextKeyTyped();
            }

            if (input == ':') {
                colon = true;
            } else if (input == 'W' || input == 'w') {
                world.updateLocation(1, 0);
                colon = false;
            } else if (input == 'A' || input == 'a') {
                world.updateLocation(0, -1);
                colon = false;
            } else if (input == 'S' || input == 's') {
                world.updateLocation(-1, 0);
                colon = false;
            } else if (input == 'D' || input == 'd') {
                world.updateLocation(0, 1);
                colon = false;
            } else if (colon && (input == 'Q' || input == 'q')) {
                saveWorld();
                System.exit(0);
            }


            ter.renderFrame(world.getWorld());

        }

    }

    private void saveWorld() {
        File file = new File("savedWorld.txt");
        PrintWriter fw;
        try {
            fw = new PrintWriter(file);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        TETile[][] w = world.getWorld();


        for (int i = 0; i < w.length; i++) {
            String line = "";
            for (int j = 0; j < w[i].length; j++) {
                line += w[i][j].character();
            }
            fw.println(line);
        }
        fw.println(world.character.x);
        fw.println(world.character.y);
        fw.close();
    }

    public World getWorld() {
        return world;
    }
}

