def truncate(f,n):
	slen = len('%.*f' % (n, f))
    return str(f)[:slen]