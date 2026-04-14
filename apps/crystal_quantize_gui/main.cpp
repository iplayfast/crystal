#include "crystal/quantize/pipeline.hpp"

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QProgressBar>
#include <QTextEdit>
#include <QCheckBox>
#include <QGroupBox>
#include <QSpinBox>
#include <QThread>
#include <QMessageBox>
#include <QFontDatabase>
#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <iostream>
#include <sstream>

struct OllamaModel {
    QString name;
    QString path;
    QString size;
    qint64 bytes;
};

class QuantizeWorker : public QThread {
    Q_OBJECT
public:
    QStringList input_models;
    QString output_path;
    QString dataset_path;
    QString keep_layers_regex;
    int num_chunks = 100;
    bool no_calibrate = false;
    bool verbose = false;

    crystal::PipelineResult result;

    void run() override {
        crystal::PipelineOptions options;
        options.output_path = output_path.toStdString();
        options.dataset_path = dataset_path.toStdString();
        options.keep_layers_regex = keep_layers_regex.toStdString();
        options.num_chunks = num_chunks;
        options.no_calibrate = no_calibrate;
        options.verbose = verbose;

        for (const auto& m : input_models) {
            options.input_models.push_back(m.toStdString());
        }

        result = crystal::run_pipeline(options);
    }
};

class CrystalQuantizeGui : public QWidget {
    Q_OBJECT
public:
    CrystalQuantizeGui() {
        setWindowTitle("Crystal Quantize - Ternary GGUF Quantizer");
        setMinimumSize(700, 650);

        discoverOllamaModels();

        auto* main_layout = new QVBoxLayout(this);

        auto* ollama_group = new QGroupBox("Ollama Models");
        auto* ollama_layout = new QVBoxLayout(ollama_group);
        
        ollama_combo = new QComboBox;
        ollama_combo->setPlaceholderText("Select an Ollama model...");
        populateOllamaCombo();
        ollama_layout->addWidget(ollama_combo);
        
        auto* ollama_button = new QPushButton("Refresh Model List");
        ollama_layout->addWidget(ollama_button);
        connect(ollama_button, &QPushButton::clicked, this, [this]() {
            discoverOllamaModels();
            populateOllamaCombo();
        });

        connect(ollama_combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &CrystalQuantizeGui::onOllamaModelSelected);

        main_layout->addWidget(ollama_group);

        auto* input_group = new QGroupBox("Or Select Custom GGUF File");
        auto* input_layout = new QVBoxLayout(input_group);
        
        input_model_edit = new QLineEdit;
        input_model_edit->setPlaceholderText("Select input GGUF model file...");
        auto* input_button = new QPushButton("Browse...");
        auto* input_layout_row = new QHBoxLayout;
        input_layout_row->addWidget(input_model_edit);
        input_layout_row->addWidget(input_button);
        input_layout->addLayout(input_layout_row);

        connect(input_button, &QPushButton::clicked, this, [this]() {
            QString file = QFileDialog::getOpenFileName(this, "Select Input GGUF Model",
                QString(), "GGUF Files (*.gguf);;All Files (*)");
            if (!file.isEmpty()) {
                input_model_edit->setText(file);
            }
        });

        add_model_button = new QPushButton("Add Another Model (for ensemble)");
        input_layout->addWidget(add_model_button);
        connect(add_model_button, &QPushButton::clicked, this, &CrystalQuantizeGui::addInputModel);

        main_layout->addWidget(input_group);

        auto* output_group = new QGroupBox("Output");
        auto* output_layout = new QVBoxLayout(output_group);
        
        output_path_edit = new QLineEdit;
        output_path_edit->setPlaceholderText("Output GGUF file path...");
        auto* output_button = new QPushButton("Browse...");
        auto* output_layout_row = new QHBoxLayout;
        output_layout_row->addWidget(output_path_edit);
        output_layout_row->addWidget(output_button);
        output_layout->addLayout(output_layout_row);

        connect(output_button, &QPushButton::clicked, this, [this]() {
            QString file = QFileDialog::getSaveFileName(this, "Select Output GGUF File",
                QString(), "GGUF Files (*.gguf);;All Files (*)");
            if (!file.isEmpty()) {
                output_path_edit->setText(file);
            }
        });

        main_layout->addWidget(output_group);

        auto* dataset_group = new QGroupBox("Calibration (Optional)");
        auto* dataset_layout = new QVBoxLayout(dataset_group);
        
        dataset_path_edit = new QLineEdit;
        dataset_path_edit->setPlaceholderText("Calibration dataset file (optional)...");
        auto* dataset_button = new QPushButton("Browse...");
        auto* dataset_layout_row = new QHBoxLayout;
        dataset_layout_row->addWidget(dataset_path_edit);
        dataset_layout_row->addWidget(dataset_button);
        dataset_layout->addLayout(dataset_layout_row);

        connect(dataset_button, &QPushButton::clicked, this, [this]() {
            QString file = QFileDialog::getOpenFileName(this, "Select Calibration Dataset",
                QString(), "Text Files (*.txt);;All Files (*)");
            if (!file.isEmpty()) {
                dataset_path_edit->setText(file);
            }
        });

        main_layout->addWidget(dataset_group);

        auto* options_group = new QGroupBox("Options");
        auto* options_layout = new QGridLayout(options_group);
        
        options_layout->addWidget(new QLabel("Keep Layers Regex:"), 0, 0);
        keep_layers_edit = new QLineEdit("embed|output");
        keep_layers_edit->setToolTip("Regex to match layer names that should be kept at F16");
        options_layout->addWidget(keep_layers_edit, 0, 1);

        options_layout->addWidget(new QLabel("Calibration Chunks:"), 1, 0);
        chunks_spin = new QSpinBox;
        chunks_spin->setRange(1, 10000);
        chunks_spin->setValue(100);
        chunks_spin->setToolTip("Number of calibration chunks for importance computation");
        options_layout->addWidget(chunks_spin, 1, 1);

        no_calibrate_check = new QCheckBox("Skip Calibration (use weight-only stats)");
        no_calibrate_check->setToolTip("Don't run calibration, use simple absmean quantization");
        options_layout->addWidget(no_calibrate_check, 2, 0, 1, 2);

        verbose_check = new QCheckBox("Verbose Output");
        verbose_check->setToolTip("Print detailed per-layer quantization statistics");
        options_layout->addWidget(verbose_check, 3, 0, 1, 2);

        main_layout->addWidget(options_group);

        auto* progress_group = new QGroupBox("Progress");
        auto* progress_layout = new QVBoxLayout(progress_group);
        
        progress_bar = new QProgressBar;
        progress_bar->setRange(0, 100);
        progress_bar->setValue(0);
        progress_layout->addWidget(progress_bar);

        log_text = new QTextEdit;
        log_text->setReadOnly(true);
        log_text->setMaximumHeight(200);
        log_text->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
        progress_layout->addWidget(log_text);

        main_layout->addWidget(progress_group);

        quantize_button = new QPushButton("Quantize");
        quantize_button->setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }");
        main_layout->addWidget(quantize_button);

        connect(quantize_button, &QPushButton::clicked, this, &CrystalQuantizeGui::startQuantize);
    }

private:
    void discoverOllamaModels() {
        ollama_models.clear();
        
        QString ollama_base = QDir::homePath() + "/.ollama/models/blobs";
        QDir dir(ollama_base);
        
        if (!dir.exists()) {
            return;
        }
        
        QFileInfoList entries = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
        
        for (const QFileInfo& info : entries) {
            qint64 size = info.size();
            QString fname = info.fileName();
            
            bool is_model_blob = (fname.startsWith("sha256:") || fname.startsWith("sha256-")) && size > 100 * 1024 * 1024;
            
            if (is_model_blob) {
                OllamaModel model;
                model.path = info.absoluteFilePath();
                model.bytes = size;
                
                if (size >= 1024ULL * 1024 * 1024) {
                    model.size = QString::number(size / (1024.0 * 1024.0 * 1024.0), 'f', 1) + " GB";
                } else {
                    model.size = QString::number(size / (1024.0 * 1024.0), 'f', 0) + " MB";
                }
                
                QString hash = fname;
                hash.replace("sha256:", "").replace("sha256-", "");
                model.name = hash.left(12) + "... (" + model.size + ")";
                
                ollama_models.append(model);
            }
        }
        
        std::sort(ollama_models.begin(), ollama_models.end(),
            [](const OllamaModel& a, const OllamaModel& b) {
                return b.bytes < a.bytes;
            });
    }

    void populateOllamaCombo() {
        ollama_combo->clear();
        
        if (ollama_models.isEmpty()) {
            ollama_combo->addItem("No Ollama models found", "");
            return;
        }
        
        ollama_combo->addItem("Select an Ollama model...", "");
        
        for (const OllamaModel& model : ollama_models) {
            ollama_combo->addItem(model.name, model.path);
        }
    }

private slots:
    void onOllamaModelSelected(int index) {
        if (index > 0) {
            QString path = ollama_combo->currentData().toString();
            if (!path.isEmpty()) {
                input_model_edit->setText(path);
            }
        }
    }

    void addInputModel() {
        QString file = QFileDialog::getOpenFileName(this, "Select Additional GGUF Model",
            QString(), "GGUF Files (*.gguf);;All Files (*)");
        if (!file.isEmpty()) {
            input_model_edit->setText(file);
            additional_models.append(file);
            log_text->append("Added model for ensemble: " + file);
        }
    }

    void startQuantize() {
        QString input_path = input_model_edit->text();
        QString output_path = output_path_edit->text();
        
        if (input_path.isEmpty()) {
            QMessageBox::warning(this, "Error", "Please select an input GGUF model");
            return;
        }
        
        if (output_path.isEmpty()) {
            QMessageBox::warning(this, "Error", "Please specify an output file path");
            return;
        }

        quantize_button->setEnabled(false);
        progress_bar->setValue(0);
        log_text->clear();
        log_text->append("Starting quantization...");
        
        worker = new QuantizeWorker;
        worker->input_models = QStringList() << input_path << additional_models;
        worker->output_path = output_path;
        worker->dataset_path = dataset_path_edit->text();
        worker->keep_layers_regex = keep_layers_edit->text();
        worker->num_chunks = chunks_spin->value();
        worker->no_calibrate = no_calibrate_check->isChecked();
        worker->verbose = verbose_check->isChecked();

        connect(worker, &QThread::started, this, [this]() {
            progress_bar->setValue(30);
            log_text->append("Loading model...");
        });
        
        connect(worker, &QThread::finished, this, &CrystalQuantizeGui::onQuantizeFinished);
        
        worker->start();
    }

    void onQuantizeFinished() {
        progress_bar->setValue(100);
        
        if (worker->result.success) {
            std::ostringstream oss;
            oss << "Quantization complete!\n"
                << "Original size: " << worker->result.original_size_bytes << " bytes\n"
                << "Quantized size: " << worker->result.quantized_size_bytes << " bytes\n"
                << "Compression: " << (worker->result.compression_ratio * 100.0) << "%\n"
                << "Tensors quantized: " << worker->result.tensors_quantized;
            log_text->append(oss.str().c_str());
            
            QMessageBox::information(this, "Success", 
                QString("Quantization complete!\n\nOutput: %1\nSize: %2 MB")
                .arg(worker->result.quantized_size_bytes / (1024.0 * 1024.0), 0, 'f', 2)
                .arg(output_path_edit->text()));
        } else {
            log_text->append(QString("Error: %1").arg(worker->result.error_message.c_str()));
            QMessageBox::critical(this, "Error", 
                QString("Quantization failed: %1").arg(worker->result.error_message.c_str()));
        }
        
        quantize_button->setEnabled(true);
        worker->deleteLater();
        worker = nullptr;
    }

private:
    QList<OllamaModel> ollama_models;
    QComboBox* ollama_combo;
    QLineEdit* input_model_edit;
    QPushButton* add_model_button;
    QStringList additional_models;
    
    QLineEdit* output_path_edit;
    QLineEdit* dataset_path_edit;
    
    QLineEdit* keep_layers_edit;
    QSpinBox* chunks_spin;
    QCheckBox* no_calibrate_check;
    QCheckBox* verbose_check;
    
    QProgressBar* progress_bar;
    QTextEdit* log_text;
    QPushButton* quantize_button;
    
    QuantizeWorker* worker = nullptr;
};

#include "main.moc"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("Crystal Quantize");
    app.setApplicationVersion("1.0");
    
    CrystalQuantizeGui gui;
    gui.show();
    
    return app.exec();
}
