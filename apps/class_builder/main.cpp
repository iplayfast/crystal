#include <QApplication>
#include <QMainWindow>
#include <QTreeWidget>
#include <QTabWidget>
#include <QComboBox>
#include <QLineEdit>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QMenuBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QLabel>
#include <QSplitter>
#include <QHeaderView>

#ifndef Q_MOC_RUN
#include <nlohmann/json.hpp>
#endif

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>

// ── Data Model ──────────────────────────────────────────────────────────────

struct MemberVar {
    std::string type;       // e.g. "int", "std::string", "crystal::FuzzySet<double>"
    std::string name;
    int array_count = 0;    // 0 = not array
    bool property = false;  // generate getter/setter
};

struct ClassDef {
    std::string class_name = "MyClass";
    std::string base_class;              // empty = no base
    std::string namespace_name;
    std::vector<MemberVar> members;

    // ── Code Generation ────────────────────────────────────────────────────

    std::string generate_hpp() const {
        std::ostringstream out;
        out << "#pragma once\n\n";

        // Includes
        out << "#include <string>\n";
        out << "#include <vector>\n";
        if (uses_crystal_type()) {
            out << "#include <crystal/fuzzy/fuzzy_set.hpp>\n";
            out << "#include <crystal/nn/backprop.hpp>\n";
            out << "#include <crystal/nn/blob_network.hpp>\n";
        }
        if (!base_class.empty() && base_class.find("crystal::") != std::string::npos) {
            // Already included above
        }
        out << "\n";

        if (!namespace_name.empty()) out << "namespace " << namespace_name << " {\n\n";

        out << "class " << class_name;
        if (!base_class.empty()) out << " : public " << base_class;
        out << " {\npublic:\n";

        // Constructor
        out << "    " << class_name << "()";
        if (!base_class.empty()) out << " : " << base_class << "()";
        out << " {}\n";
        out << "    ~" << class_name << "() = default;\n\n";

        // Getters/setters for property members
        for (auto& m : members) {
            if (m.property) {
                std::string full_type = member_type_str(m);
                out << "    [[nodiscard]] const " << full_type << "& get_" << m.name
                    << "() const { return " << m.name << "_; }\n";
                out << "    void set_" << m.name << "(const " << full_type << "& v) { "
                    << m.name << "_ = v; }\n\n";
            }
        }

        out << "private:\n";
        for (auto& m : members) {
            std::string suffix = m.property ? "_" : "";
            if (m.array_count > 0) {
                out << "    std::vector<" << m.type << "> " << m.name << suffix
                    << " = std::vector<" << m.type << ">(" << m.array_count << ");\n";
            } else {
                out << "    " << m.type << " " << m.name << suffix << "{};\n";
            }
        }

        out << "};\n";
        if (!namespace_name.empty()) out << "\n} // namespace " << namespace_name << "\n";
        return out.str();
    }

    std::string generate_cpp() const {
        std::ostringstream out;
        out << "#include \"" << class_name << ".hpp\"\n\n";
        if (!namespace_name.empty()) out << "namespace " << namespace_name << " {\n\n";
        out << "// " << class_name << " implementation\n\n";
        if (!namespace_name.empty()) out << "} // namespace " << namespace_name << "\n";
        return out.str();
    }

    nlohmann::json to_json() const {
        nlohmann::json j;
        j["class_name"] = class_name;
        j["base_class"] = base_class;
        j["namespace"] = namespace_name;
        j["members"] = nlohmann::json::array();
        for (auto& m : members) {
            j["members"].push_back({
                {"type", m.type}, {"name", m.name},
                {"array_count", m.array_count}, {"property", m.property}
            });
        }
        return j;
    }

    static ClassDef from_json(const nlohmann::json& j) {
        ClassDef d;
        d.class_name = j.value("class_name", "MyClass");
        d.base_class = j.value("base_class", "");
        d.namespace_name = j.value("namespace", "");
        for (auto& m : j.at("members")) {
            d.members.push_back({
                m.at("type").get<std::string>(),
                m.at("name").get<std::string>(),
                m.value("array_count", 0),
                m.value("property", false)
            });
        }
        return d;
    }

private:
    bool uses_crystal_type() const {
        auto has = [](const std::string& s) { return s.find("crystal::") != std::string::npos; };
        if (has(base_class)) return true;
        for (auto& m : members) if (has(m.type)) return true;
        return false;
    }

    static std::string member_type_str(const MemberVar& m) {
        if (m.array_count > 0) return "std::vector<" + m.type + ">";
        return m.type;
    }
};

// ── ClassBuilderWindow ─────────────────────────────────────────────────────

class ClassBuilderWindow : public QMainWindow {
    Q_OBJECT
public:
    ClassDef def;
    QTreeWidget* tree;
    QLineEdit* class_name_edit;
    QLineEdit* namespace_edit;
    QComboBox* base_combo;
    QComboBox* var_type_combo;
    QLineEdit* var_name_edit;
    QSpinBox* var_array_spin;
    QCheckBox* var_property_check;
    QLabel* preview_label;

    ClassBuilderWindow() {
        setWindowTitle("Crystal ClassBuilder");
        resize(800, 600);

        // Menu
        auto* file_menu = menuBar()->addMenu("&File");
        file_menu->addAction("&New", QKeySequence::New, this, &ClassBuilderWindow::new_class);
        file_menu->addAction("&Open JSON...", QKeySequence::Open, this, &ClassBuilderWindow::open_json);
        file_menu->addAction("&Save JSON...", this, &ClassBuilderWindow::save_json);
        file_menu->addSeparator();
        file_menu->addAction("Export &.hpp...", this, &ClassBuilderWindow::export_hpp);
        file_menu->addAction("Export .&cpp...", this, &ClassBuilderWindow::export_cpp);
        file_menu->addSeparator();
        file_menu->addAction("&Quit", QKeySequence::Quit, this, &QWidget::close);

        // Splitter: tree on left, tabs on right
        auto* splitter = new QSplitter(Qt::Horizontal);

        tree = new QTreeWidget;
        tree->setHeaderLabels({"Element", "Detail"});
        tree->header()->setStretchLastSection(true);
        splitter->addWidget(tree);

        // Right side tabs
        auto* tabs = new QTabWidget;

        // Tab 1: Class Settings
        auto* class_tab = new QWidget;
        auto* class_layout = new QFormLayout(class_tab);

        class_name_edit = new QLineEdit("MyClass");
        connect(class_name_edit, &QLineEdit::textChanged, this, &ClassBuilderWindow::on_settings_changed);
        class_layout->addRow("Class Name:", class_name_edit);

        namespace_edit = new QLineEdit;
        connect(namespace_edit, &QLineEdit::textChanged, this, &ClassBuilderWindow::on_settings_changed);
        class_layout->addRow("Namespace:", namespace_edit);

        base_combo = new QComboBox;
        base_combo->setEditable(true);
        base_combo->addItems({
            "(none)",
            "crystal::FuzzySet<double>",
            "crystal::FuzzySet<float>",
            "crystal::BackpropNetwork<double>",
            "crystal::BackpropNetwork<float>",
            "crystal::BlobNetwork"
        });
        connect(base_combo, &QComboBox::currentTextChanged, this, &ClassBuilderWindow::on_settings_changed);
        class_layout->addRow("Base Class:", base_combo);

        tabs->addTab(class_tab, "Class Settings");

        // Tab 2: Add Variable
        auto* var_tab = new QWidget;
        auto* var_layout = new QFormLayout(var_tab);

        var_type_combo = new QComboBox;
        var_type_combo->setEditable(true);
        var_type_combo->addItems({
            "bool", "char", "int", "float", "double", "std::string",
            "crystal::FuzzySet<double>", "crystal::FuzzySet<float>",
            "crystal::BackpropNetwork<double>", "crystal::BlobNetwork"
        });
        var_layout->addRow("Type:", var_type_combo);

        var_name_edit = new QLineEdit;
        var_layout->addRow("Name:", var_name_edit);

        var_array_spin = new QSpinBox;
        var_array_spin->setRange(0, 10000);
        var_array_spin->setSpecialValueText("Not array");
        var_layout->addRow("Array Count:", var_array_spin);

        var_property_check = new QCheckBox("Generate getter/setter");
        var_layout->addRow(var_property_check);

        auto* add_var_btn = new QPushButton("Add Variable");
        connect(add_var_btn, &QPushButton::clicked, this, &ClassBuilderWindow::add_variable);
        var_layout->addRow(add_var_btn);

        tabs->addTab(var_tab, "Add Variable");

        // Tab 3: Remove Variable
        auto* rm_tab = new QWidget;
        auto* rm_layout = new QVBoxLayout(rm_tab);
        auto* rm_btn = new QPushButton("Remove Selected Variable");
        connect(rm_btn, &QPushButton::clicked, this, &ClassBuilderWindow::remove_variable);
        rm_layout->addWidget(rm_btn);
        rm_layout->addStretch();
        tabs->addTab(rm_tab, "Remove Variable");

        splitter->addWidget(tabs);
        splitter->setStretchFactor(0, 2);
        splitter->setStretchFactor(1, 1);
        setCentralWidget(splitter);

        preview_label = new QLabel;
        statusBar()->addWidget(preview_label);

        refresh();
    }

private slots:
    void new_class() {
        def = ClassDef{};
        class_name_edit->setText("MyClass");
        namespace_edit->clear();
        base_combo->setCurrentIndex(0);
        refresh();
    }

    void open_json() {
        QString path = QFileDialog::getOpenFileName(this, "Open Class Definition", {},
            "JSON Files (*.json);;All Files (*)");
        if (path.isEmpty()) return;
        std::ifstream f(path.toStdString());
        nlohmann::json j;
        f >> j;
        def = ClassDef::from_json(j);
        class_name_edit->setText(QString::fromStdString(def.class_name));
        namespace_edit->setText(QString::fromStdString(def.namespace_name));
        if (!def.base_class.empty()) {
            int idx = base_combo->findText(QString::fromStdString(def.base_class));
            if (idx >= 0) base_combo->setCurrentIndex(idx);
            else base_combo->setEditText(QString::fromStdString(def.base_class));
        } else {
            base_combo->setCurrentIndex(0);
        }
        refresh();
    }

    void save_json() {
        sync_settings();
        QString path = QFileDialog::getSaveFileName(this, "Save Class Definition", {},
            "JSON Files (*.json);;All Files (*)");
        if (path.isEmpty()) return;
        std::ofstream f(path.toStdString());
        f << def.to_json().dump(2);
    }

    void export_hpp() {
        sync_settings();
        QString path = QFileDialog::getSaveFileName(this, "Export Header",
            QString::fromStdString(def.class_name) + ".hpp",
            "C++ Headers (*.hpp *.h);;All Files (*)");
        if (path.isEmpty()) return;
        std::ofstream f(path.toStdString());
        f << def.generate_hpp();
        statusBar()->showMessage("Exported: " + path, 5000);
    }

    void export_cpp() {
        sync_settings();
        QString path = QFileDialog::getSaveFileName(this, "Export Source",
            QString::fromStdString(def.class_name) + ".cpp",
            "C++ Sources (*.cpp);;All Files (*)");
        if (path.isEmpty()) return;
        std::ofstream f(path.toStdString());
        f << def.generate_cpp();
        statusBar()->showMessage("Exported: " + path, 5000);
    }

    void add_variable() {
        sync_settings();
        MemberVar m;
        m.type = var_type_combo->currentText().toStdString();
        m.name = var_name_edit->text().toStdString();
        if (m.name.empty()) {
            QMessageBox::warning(this, "Error", "Variable name is required");
            return;
        }
        m.array_count = var_array_spin->value();
        m.property = var_property_check->isChecked();
        def.members.push_back(std::move(m));
        var_name_edit->clear();
        refresh();
    }

    void remove_variable() {
        auto* item = tree->currentItem();
        if (!item || !item->parent()) return;
        int idx = item->parent()->indexOfChild(item);
        if (idx >= 0 && idx < (int)def.members.size()) {
            def.members.erase(def.members.begin() + idx);
            refresh();
        }
    }

    void on_settings_changed() {
        sync_settings();
        refresh();
    }

private:
    void sync_settings() {
        def.class_name = class_name_edit->text().toStdString();
        def.namespace_name = namespace_edit->text().toStdString();
        QString base = base_combo->currentText();
        def.base_class = (base == "(none)") ? "" : base.toStdString();
    }

    void refresh() {
        tree->clear();
        auto* root = new QTreeWidgetItem(tree, {
            QString::fromStdString("class " + def.class_name),
            QString::fromStdString(def.base_class.empty() ? "" : ": " + def.base_class)
        });

        auto* members_node = new QTreeWidgetItem(root, {"Members", ""});
        for (auto& m : def.members) {
            QString detail = QString::fromStdString(m.type);
            if (m.array_count > 0) detail += "[" + QString::number(m.array_count) + "]";
            if (m.property) detail += " [property]";
            new QTreeWidgetItem(members_node, {QString::fromStdString(m.name), detail});
        }

        tree->expandAll();
        preview_label->setText(QString("class %1 : %2 members")
            .arg(QString::fromStdString(def.class_name))
            .arg(def.members.size()));
    }
};

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("Crystal ClassBuilder");

    ClassBuilderWindow w;
    w.show();
    return app.exec();
}

#include "main.moc"
