package com.example.petidentifier;

import static android.provider.MediaStore.Images.Media.getBitmap;

import static org.pytorch.Module.load;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import org.pytorch.IValue;

import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.LiteModuleLoader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


public class MainActivity extends AppCompatActivity {
    private static final int STORAGE_PERMISSION_CODE = 113;
    Uri selectedImage;
    ImageView imageView;
    Button loadImage;
    Button classifyBtn;
    Bitmap image;
    Bitmap bitmap;
    TextView textView;
    Module module = null;
    String className = "";
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        classifyBtn = findViewById(R.id.classifyBtn);
        imageView = findViewById(R.id.imageView);
        loadImage = findViewById(R.id.loadImage);
        textView = findViewById(R.id.textView);
//        try {
//            //image = getBitmap(getApplicationContext().getContentResolver(), selectedImage);
//            //bitmap = BitmapFactory.decodeStream(image);
//            bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
//            module = LiteModuleLoader.load(assetFilePath(this, "model.ptl"));
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
//                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
//
//        // running the model
//        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
//
//        // getting tensor content as java array of floats
//        final float[] scores = outputTensor.getDataAsFloatArray();
//
//        // searching for the index with maximum score
//        float maxScore = -Float.MAX_VALUE;
//        int maxScoreIdx = -1;
//        for (int i = 0; i < scores.length; i++) {
//            if (scores[i] > maxScore) {
//                maxScore = scores[i];
//                maxScoreIdx = i;
//            }
//        }
//        className = com.example.individualproject.ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
//        textView.setText(className);

        loadImage.setOnClickListener(new View.OnClickListener() {

            public void onClick(View view){

                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                checkPermission(Manifest.permission.READ_EXTERNAL_STORAGE, STORAGE_PERMISSION_CODE);
                startActivityForResult(intent, 3);
            }
        });

        classifyBtn.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {

                Bitmap bitmap = null;
                Module module = null;

                //Getting the image from the image view
                ImageView imageView = (ImageView) findViewById(R.id.image);

                try {
                    //Read the image as Bitmap
                    bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), selectedImage);;

                    //Here we reshape the image into 400*400
                    bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);

                    //Loading the model file.
                    module = LiteModuleLoader.load(assetFilePath(MainActivity.this, "model.ptl"));
                } catch (IOException e) {
                    finish();
                }

                //Input Tensor
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                        bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );

                //Calling the forward of the model to run our input
                final Tensor output = module.forward(IValue.from(input)).toTensor();
                final float[] score_arr = output.getDataAsFloatArray();

                // Fetch the index of the value with maximum score
                float max_score = -Float.MAX_VALUE;
                int ms_ix = -1;
                for (int i = 0; i < score_arr.length; i++) {
                    if (score_arr[i] > max_score) {
                        max_score = score_arr[i];
                        ms_ix = i;
                    }
                }

                //Fetching the name from the list based on the index
                String detected_class = com.example.petidentifier.ImageNetClasses.IMAGENET_CLASSES[ms_ix];

                //Writing the detected class in to the text view of the layout
                textView.setText(detected_class);
            }
        });
    }

    public void checkPermission(String permission, int requestCode){
        if(ContextCompat.checkSelfPermission(MainActivity.this, permission)== PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission},requestCode);
        }else{
            Toast.makeText(MainActivity.this, "Permission already Granted", Toast.LENGTH_SHORT);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if(requestCode==STORAGE_PERMISSION_CODE){
            if(grantResults.length> 0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
                loadImage.setText("Permission Granted");
                Toast.makeText(this, "Storage permission Granted", Toast.LENGTH_SHORT).show();
            }else{
                Toast.makeText(this, "Storage permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            selectedImage = data.getData();
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageURI(selectedImage);
        }
    }
<<<<<<< Updated upstream
    
=======

>>>>>>> Stashed changes
    public HashMap<Integer, String> buildHashMapFromFile(){
        HashMap<Integer, String> map = new HashMap<>();
        BufferedReader objReader;
        try {
            InputStream is = getAssets().open("imagenet_classes.txt");
            objReader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            String line;
            while ((line = objReader.readLine()) != null){
                String[] classInfo = line.split(", ",2);
                String number = classInfo[0].trim();
                String className = classInfo[1].trim();

                if (!number.equals("") && !className.equals("")){
                    map.put(Integer.parseInt(number), className);
                }
            }
            objReader.close();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return map;
<<<<<<< Updated upstream
    }	
	
=======
    }
>>>>>>> Stashed changes

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}