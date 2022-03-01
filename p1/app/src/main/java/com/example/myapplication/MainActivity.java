package com.example.myapplication;

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
        classifyBtn = findViewById(R.id.negative);
        imageView = findViewById(R.id.imageView);
        loadImage = findViewById(R.id.gallery);
        textView = findViewById(R.id.result_text);


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
                ImageView imageView = (ImageView) findViewById(R.id.image);

                try {

                    bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), selectedImage);;
                    bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);
                    module = LiteModuleLoader.load(assetFilePath(MainActivity.this, "model.ptl"));
                } catch (IOException e) {
                    finish();
                }
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                        bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );
                final Tensor output = module.forward(IValue.from(input)).toTensor();


                final float[] score_arr = output.getDataAsFloatArray();
                float max_score = -Float.MAX_VALUE;
                int ms_ix = -1;
                for (int i = 0; i < score_arr.length; i++) {
                    if (score_arr[i] > max_score) {
                        max_score = score_arr[i];
                        ms_ix = i;
                    }
                }


                String detected_class = com.example.myapplication.ImageNetClasses.IMAGENET_CLASSES[ms_ix];

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