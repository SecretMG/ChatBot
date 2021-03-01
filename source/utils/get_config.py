import configparser

def get_config(config_file='config.ini'):
    # 默认在工程文件下调用该函数
    parser = configparser.ConfigParser()
    parser.read(config_file, encoding='utf-8')    # 从配置文件中读参数
    string_configs = [(k, v) for k, v in parser.items('string')]
    int_configs = [(k, int(v)) for k, v in parser.items('int')]
    float_configs = [(k, float(v)) for k, v in parser.items('float')]
    return dict(string_configs + int_configs + float_configs)


if __name__ == '__main__':
    configs = get_config('../config.ini')
    print(configs)